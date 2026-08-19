import { normalizeUrl } from "./url-normalizer.js";
import { withDeadline } from "./timeout.js";

// r.jina.ai had no client-side deadline: a hung read held the Worker invocation
// open until the platform killed it, taking the sibling reads down with it.
const READ_REQUEST_TIMEOUT_MS = 30000;

/** Passage size for question-grounded reads, in words (CJK: characters). */
export const DEFAULT_SNIPPET_TOKENS = 100;
/** Passages returned for question-grounded reads. */
export const DEFAULT_SNIPPET_TOPK = 1;

// ============================================================================
// TYPES AND INTERFACES
// ============================================================================

export interface ReadUrlConfig {
    url: string;
    withAllLinks?: boolean;
    withAllImages?: boolean;
    /**
     * When set, the page is reduced to the passage(s) that best answer this
     * question, by the same read -> chunk -> rerank pipeline that backs
     * search_web_deep. Absent (the default), the full content is returned
     * exactly as before.
     */
    question?: string;
    /** Passage size for question-grounded reads, in words (CJK: characters). Default 100. */
    tokens?: number;
    /** Number of passages to keep for question-grounded reads. Default 1. */
    topk?: number;
}

export interface ReadUrlResult {
    success: boolean;
    url: string;
    structuredData: any;
    withAllLinks: boolean;
    withAllImages: boolean;
}

export interface ReadUrlError {
    error: string;
    url: string;
}

export type ReadUrlResponse = ReadUrlResult | ReadUrlError;

// ============================================================================
// QUESTION-GROUNDED READ
// ============================================================================

/**
 * Read a page and return only the passages answering `question`.
 *
 * Delegates to svip.jina.ai's `url` + `q` form rather than chunking and ranking
 * here. That endpoint runs the same code as search_web_deep, so a passage means
 * the same thing from either tool, and its chunker does markdown-aware work this
 * Worker has no business duplicating: it strips code blocks, tables and nav
 * furniture before splitting, which is the difference between returning a bare
 * `## Heading` and returning the paragraph under it. Doing it server-side also
 * keeps the reranker call off the Worker's CPU budget.
 */
async function readWithQuestion(
    normalizedUrl: string,
    urlConfig: ReadUrlConfig,
    bearerToken?: string
): Promise<{ snippets: string[]; title?: string; } | null> {
    const response = await fetch('https://svip.jina.ai/', {
        method: 'POST',
        signal: AbortSignal.timeout(READ_REQUEST_TIMEOUT_MS),
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            ...(bearerToken ? { 'Authorization': `Bearer ${bearerToken}` } : {}),
        },
        body: JSON.stringify({
            q: urlConfig.question,
            url: normalizedUrl,
            topk: urlConfig.topk ?? DEFAULT_SNIPPET_TOPK,
            // The MCP surface calls this `tokens`; server-side the unit is words
            // (CJK: characters), which is what its sentence-boundary chunker
            // actually measures.
            chunk_size: urlConfig.tokens ?? DEFAULT_SNIPPET_TOKENS,
        }),
    });

    if (!response.ok) return null;

    const data = await response.json() as { results?: Array<{ snippet?: string; title?: string; }>; };
    const first = data?.results?.[0];
    if (!first?.snippet) return null;

    // One url in, so at most one result out; its snippet already joins the top
    // `topk` passages with the separator deep search uses.
    return { snippets: [first.snippet], title: first.title };
}

// ============================================================================
// CORE URL READING LOGIC
// ============================================================================

/**
 * Core function to read and extract content from a URL
 */
export async function readUrlFromConfig(
    urlConfig: ReadUrlConfig,
    bearerToken?: string
): Promise<ReadUrlResponse> {
    try {
        // Normalize the URL first
        const normalizedUrl = normalizeUrl(urlConfig.url);
        if (!normalizedUrl) {
            return { error: "Invalid or unsupported URL", url: urlConfig.url };
        }

        const headers: Record<string, string> = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'X-Md-Link-Style': 'discarded',
        };

        // Add Authorization header if bearer token is available
        if (bearerToken) {
            headers['Authorization'] = `Bearer ${bearerToken}`;
        }

        if (urlConfig.withAllLinks) {
            headers['X-With-Links-Summary'] = 'all';
        }

        if (urlConfig.withAllImages) {
            headers['X-With-Images-Summary'] = 'true';
        } else {
            headers['X-Retain-Images'] = 'none';
        }

        // With a question, the passage extraction runs concurrently with the
        // plain read. The read is not wasted work: it is the fallback body if
        // extraction comes back empty, and it is what supplies links/images,
        // which the passage endpoint does not return. Sequencing them would add a
        // full round-trip to every question-grounded read for no benefit.
        const question = urlConfig.question?.trim();

        const [response, snippetResult] = await Promise.all([
            fetch('https://r.jina.ai/', {
                method: 'POST',
                signal: AbortSignal.timeout(READ_REQUEST_TIMEOUT_MS),
                headers,
                body: JSON.stringify({ url: normalizedUrl }),
            }),
            question
                // A failure here must not fail the read; it degrades to full content.
                ? readWithQuestion(normalizedUrl, urlConfig, bearerToken).catch(() => null)
                : Promise.resolve(null),
        ]);

        if (!response.ok) {
            return { error: `HTTP ${response.status}: ${response.statusText}`, url: urlConfig.url };
        }

        const data = await response.json() as any;

        if (!data.data) {
            return { error: "Invalid response data from r.jina.ai", url: urlConfig.url };
        }

        // Prepare structured data
        const structuredData: any = {
            url: data.data.url,
            title: data.data.title,
        };

        if (urlConfig.withAllLinks && data.data.links) {
            structuredData.links = data.data.links.map((link: [string, string]) => ({
                anchorText: link[0],
                url: link[1]
            }));
        }

        if (urlConfig.withAllImages && data.data.images) {
            structuredData.images = data.data.images;
        }

        const content: string = data.data.content || "";

        if (question) {
            if (snippetResult) {
                structuredData.question = question;
                structuredData.snippets = snippetResult.snippets;
                // The full body is deliberately omitted: the point of passing a
                // question is to not carry the whole page. `snippet_source`
                // mirrors the field search_web_deep sets, so a caller can tell a
                // ranked passage from a fallback full read.
                structuredData.snippet_source = 'content';

                return {
                    success: true,
                    url: urlConfig.url,
                    structuredData,
                    withAllLinks: urlConfig.withAllLinks || false,
                    withAllImages: urlConfig.withAllImages || false
                };
            }

            // Extraction could not run, or ran and found nothing rankable.
            // Returning the full page is strictly more useful than an error, but
            // say so rather than letting the caller believe it is looking at a
            // ranked passage.
            structuredData.question = question;
            structuredData.snippet_source = 'full_content';
            structuredData.note = 'Question-grounded extraction was unavailable for this page; returning full content.';
        }

        structuredData.content = content;

        return {
            success: true,
            url: urlConfig.url,
            structuredData,
            withAllLinks: urlConfig.withAllLinks || false,
            withAllImages: urlConfig.withAllImages || false
        };
    } catch (error) {
        return {
            error: error instanceof Error ? error.message : String(error),
            url: urlConfig.url
        };
    }
}

/**
 * Execute multiple URL reads in parallel with timeout
 */
export async function executeParallelUrlReads(
    urlConfigs: ReadUrlConfig[],
    bearerToken?: string,
    timeout: number = 30000
): Promise<ReadUrlResponse[]> {
    // Per-URL deadline. The whole batch used to be raced against one rejecting
    // timeout, so one slow page threw away every read that had already
    // succeeded and the caller got a single generic error instead.
    return Promise.all(
        urlConfigs.map((urlConfig) =>
            withDeadline<ReadUrlResponse>(
                () => readUrlFromConfig(urlConfig, bearerToken),
                timeout,
                () => ({ error: `Read timed out after ${timeout}ms`, url: urlConfig.url })
            )
        )
    );
}
