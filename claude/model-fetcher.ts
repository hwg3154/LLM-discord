/**
 * Dynamic Model Fetcher
 * 
 * Discovers available Claude models using two strategies:
 * 1. Primary: Anthropic REST API (requires ANTHROPIC_API_KEY)
 * 2. Fallback: Parse model IDs from the installed Claude CLI binary
 * 
 * Both strategies auto-refresh periodically and fall back to
 * hardcoded defaults when neither source is available.
 */

import type { ModelInfo } from "./enhanced-client.ts";

// Normalized model entry used internally.
interface AnthropicModelEntry {
  id: string;
  display_name: string;
  created_at: string;
  type: string;
  owned_by?: string;
}

/**
 * A raw entry from a /v1/models response. Anthropic and OpenAI-compatible
 * gateways (Open WebUI, LiteLLM, vLLM, Ollama proxies) describe the same
 * thing with different field names, so both shapes are accepted.
 */
interface RawModelEntry {
  id: string;
  // Anthropic shape
  display_name?: string;
  created_at?: string;
  type?: string;
  // OpenAI / Open WebUI shape
  name?: string;
  created?: number;
  object?: string;
  owned_by?: string;
}

interface AnthropicModelsResponse {
  data: RawModelEntry[];
  has_more?: boolean;
  first_id?: string;
  last_id?: string;
}

/** True when the base URL is the official Anthropic API. */
function isOfficialAnthropic(baseUrl: string): boolean {
  try {
    return new URL(baseUrl).hostname === "api.anthropic.com";
  } catch {
    return false;
  }
}

/**
 * Coerce either response shape into a single internal representation.
 * OpenAI-style endpoints use name/created/object where Anthropic uses
 * display_name/created_at/type.
 */
function normalizeEntry(raw: RawModelEntry): AnthropicModelEntry {
  let createdAt = raw.created_at ?? "";
  if (!createdAt && typeof raw.created === "number" && raw.created > 0) {
    createdAt = new Date(raw.created * 1000).toISOString();
  }

  return {
    id: raw.id,
    display_name: raw.display_name || raw.name || raw.id,
    created_at: createdAt,
    type: raw.type || raw.object || "model",
    owned_by: raw.owned_by,
  };
}

/** Parse a timestamp for sorting; unknown/invalid dates sort last. */
function timestamp(value: string): number {
  const parsed = new Date(value).getTime();
  return Number.isNaN(parsed) ? 0 : parsed;
}

// Cache state
let cachedModels: Record<string, ModelInfo> | null = null;
let lastFetchTime = 0;
const CACHE_TTL_MS = 60 * 60 * 1000; // 1 hour
let refreshInterval: ReturnType<typeof setInterval> | null = null;

/**
 * Classify a model ID into a tier based on naming conventions.
 */
function classifyTier(id: string): ModelInfo['tier'] {
  if (id.includes('opus')) return 'flagship';
  if (id.includes('haiku')) return 'fast';
  if (id.includes('sonnet')) return 'balanced';
  // Legacy / Claude 3 models
  if (id.startsWith('claude-3-') && !id.startsWith('claude-3-5-')) return 'legacy';
  return 'balanced';
}

/**
 * Determine if a model ID supports extended thinking (heuristic).
 */
function inferSupportsThinking(id: string): boolean {
  // Opus and Sonnet 4+ support thinking; Haiku and legacy do not
  if (id.includes('opus')) return true;
  if (id.includes('sonnet')) {
    // Sonnet 4+ supports thinking
    const versionMatch = id.match(/sonnet-(\d+)/);
    if (versionMatch && parseInt(versionMatch[1]) >= 4) return true;
  }
  return false;
}

/**
 * Determine if a model is deprecated based on its ID.
 */
function inferDeprecated(id: string): boolean {
  // Claude 3.x (non-3.5) are deprecated
  if (id.startsWith('claude-3-') && !id.startsWith('claude-3-5-')) return true;
  // Claude 3.5 models are older generation
  if (id.startsWith('claude-3-5-')) return true;
  return false;
}

/**
 * Extract a context window size from model ID (heuristic).
 * All current Claude models use 200k context.
 */
function inferContextWindow(_id: string): number {
  return 200_000;
}

/**
 * Convert a model entry into our ModelInfo format.
 *
 * `sourceLabel` names where the entry came from. For non-Claude models served
 * by a custom gateway the Claude-specific heuristics below don't apply, so the
 * conservative defaults (no thinking, not deprecated) are used instead.
 */
function toModelInfo(entry: AnthropicModelEntry, sourceLabel: string): ModelInfo {
  const label = entry.display_name || entry.id;
  const isClaude = entry.id.startsWith("claude-");
  const owner = entry.owned_by ? `${entry.owned_by} · ` : "";

  return {
    name: label,
    description: `${owner}${entry.id} (via ${sourceLabel})`,
    contextWindow: inferContextWindow(entry.id),
    recommended: false,
    supportsThinking: isClaude ? inferSupportsThinking(entry.id) : false,
    tier: isClaude ? classifyTier(entry.id) : "balanced",
    deprecated: isClaude ? inferDeprecated(entry.id) : false,
  };
}

/**
 * Build alias entries that point to the latest model in each family.
 * Families: opus, sonnet, haiku.
 */
function buildAliases(models: Record<string, ModelInfo>, apiModels: AnthropicModelEntry[]): void {
  const families = ['opus', 'sonnet', 'haiku'] as const;
  
  for (const family of families) {
    // Find all API models in this family, sorted by created_at descending
    const familyModels = apiModels
      .filter(m => m.id.includes(family) && m.type === 'model')
      .sort((a, b) => timestamp(b.created_at) - timestamp(a.created_at));
    
    if (familyModels.length > 0) {
      const latest = familyModels[0];
      const tier = classifyTier(latest.id);
      
      models[family] = {
        name: `Claude ${family.charAt(0).toUpperCase() + family.slice(1)} (Latest)`,
        description: `Auto-resolves to latest ${family.charAt(0).toUpperCase() + family.slice(1)} via CLI alias`,
        contextWindow: inferContextWindow(latest.id),
        recommended: true,
        supportsThinking: inferSupportsThinking(latest.id),
        tier,
        aliasFor: latest.id,
      };

      // Mark the latest model as recommended
      if (models[latest.id]) {
        models[latest.id].recommended = true;
      }
    }
  }
}

/**
 * Fetch models from the Anthropic API or a custom ANTHROPIC_BASE_URL.
 * Returns null if no API key is set or the request fails.
 *
 * Strategy:
 * 1. If ANTHROPIC_BASE_URL is set, try that endpoint first (for custom proxies/endpoints)
 * 2. Fall back to the official Anthropic API (https://api.anthropic.com)
 * 3. Both endpoints must have ANTHROPIC_API_KEY set
 */
async function fetchFromAPI(): Promise<Record<string, ModelInfo> | null> {
  const apiKey = Deno.env.get("ANTHROPIC_API_KEY");
  if (!apiKey) {
    return null;
  }

  // Try custom base URL first (if configured)
  const customBaseUrl = Deno.env.get("ANTHROPIC_BASE_URL");
  const urlsToTry = customBaseUrl
    ? [customBaseUrl.replace(/\/+$/, ''), "https://api.anthropic.com"]
    : ["https://api.anthropic.com"];

  for (const baseUrl of urlsToTry) {
    try {
      const rawModels: RawModelEntry[] = [];
      let hasMore = true;
      let afterId: string | undefined;

      while (hasMore) {
        const url = new URL(`${baseUrl}/v1/models`);
        url.searchParams.set("limit", "100");
        if (afterId) {
          url.searchParams.set("after_id", afterId);
        }

        const response = await fetch(url.toString(), {
          method: "GET",
          headers: {
            "x-api-key": apiKey,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
          },
        });

        if (!response.ok) {
          console.error(`Models API error (${baseUrl}): ${response.status} ${response.statusText}`);
          const body = await response.text().catch(() => "");
          console.error(`Response body: ${body}`);
          // Try next URL if available
          break;
        }

        const data: AnthropicModelsResponse = await response.json();
        rawModels.push(...(data.data ?? []));

        // OpenAI-compatible gateways return the full list unpaginated and omit
        // has_more entirely, so an absent flag must terminate the loop.
        hasMore = data.has_more === true;
        if (hasMore && data.last_id) {
          afterId = data.last_id;
        } else {
          hasMore = false;
        }
      }

      if (rawModels.length === 0) continue;

      const isOfficial = isOfficialAnthropic(baseUrl);
      const sourceLabel = isOfficial ? "Anthropic API" : new URL(baseUrl).host;
      const entries = rawModels.map(normalizeEntry);
      const models = buildModelsFromAPI(entries, isOfficial, sourceLabel);
      const count = Object.keys(models).length;

      if (count > 0) {
        console.log(`Model fetcher: Loaded ${count} models from ${baseUrl}`);
        return models;
      }

      // Reached the endpoint but nothing survived filtering. Don't cache an
      // empty catalog — fall through to the next source.
      console.warn(
        `Model fetcher: ${rawModels.length} entries from ${baseUrl} yielded no usable models — trying next source`,
      );
    } catch (error) {
      console.error(`Failed to fetch models from ${baseUrl}:`, error instanceof Error ? error.message : String(error));
      // Try next URL if available
      continue;
    }
  }

  return null;
}

/**
 * Build the models record from API data.
 *
 * The official Anthropic API only ever serves Claude models, so anything else
 * there is junk and gets dropped. A custom ANTHROPIC_BASE_URL is an arbitrary
 * gateway that may serve any model (Qwen, Ministral, Llama, ...), so its
 * entries are kept as-is — filtering them to `claude-` would discard the whole
 * catalog and leave the picker empty.
 */
function buildModelsFromAPI(
  apiModels: AnthropicModelEntry[],
  isOfficial: boolean,
  sourceLabel: string,
): Record<string, ModelInfo> {
  const models: Record<string, ModelInfo> = {};

  const usable = isOfficial
    ? apiModels.filter(m => m.id.startsWith("claude-"))
    : apiModels.filter(m => typeof m.id === "string" && m.id.length > 0);

  for (const entry of usable) {
    models[entry.id] = toModelInfo(entry, sourceLabel);
  }

  // Convenience aliases (opus, sonnet, haiku) — only meaningful if the source
  // actually serves those families; a no-op otherwise.
  buildAliases(models, usable);

  return models;
}

/**
 * Parse model IDs from the installed Claude CLI binary.
 * Looks for patterns like "claude-xxx-yyy-YYYYMMDD" in cli.js.
 * Checks both the old package (@anthropic-ai/claude-code) and
 * the new SDK package (@anthropic-ai/claude-agent-sdk).
 */
async function parseModelsFromCLI(): Promise<string[] | null> {
  try {
    // Common install paths for Claude CLI (both old and new package names)
    const packageNames = [
      '@anthropic-ai/claude-code',
      '@anthropic-ai/claude-agent-sdk',
    ];
    const possiblePaths: string[] = [];
    
    for (const pkg of packageNames) {
      // npm global (Windows)
      const appData = Deno.env.get("APPDATA");
      if (appData) possiblePaths.push(`${appData}/npm/node_modules/${pkg}/cli.js`);
      // npm global (Unix)
      possiblePaths.push(`/usr/local/lib/node_modules/${pkg}/cli.js`);
      possiblePaths.push(`/usr/lib/node_modules/${pkg}/cli.js`);
      // User-specific npm (Unix)
      const home = Deno.env.get("HOME");
      if (home) possiblePaths.push(`${home}/.npm-global/lib/node_modules/${pkg}/cli.js`);
    }

    let cliContent: string | null = null;

    for (const cliPath of possiblePaths) {
      if (!cliPath) continue;
      try {
        cliContent = await Deno.readTextFile(cliPath);
        console.log(`Model fetcher: Found CLI binary at ${cliPath}`);
        break;
      } catch {
        // Try next path
      }
    }

    // Also try locating via `which claude` / `where claude`
    if (!cliContent) {
      try {
        const isWindows = Deno.build.os === 'windows';
        const whichCmd = isWindows ? 'where' : 'which';
        const cmd = new Deno.Command(whichCmd, {
          args: ['claude'],
          stdout: 'piped',
          stderr: 'piped',
        });
        const { stdout } = await cmd.output();
        const claudePath = new TextDecoder().decode(stdout).trim().split('\n')[0].trim();
        
        if (claudePath) {
          // Claude is a JS script — the actual CLI is in the same package
          // Resolve to the package's cli.js (check both old and new package names)
          const basePath = claudePath.replace(/[/\\]claude(\.cmd|\.ps1)?$/i, '');
          const possibleCliJsPaths = [
            `${basePath}/node_modules/@anthropic-ai/claude-code/cli.js`,
            `${basePath}/node_modules/@anthropic-ai/claude-agent-sdk/cli.js`,
          ];
          
          for (const possibleCliJs of possibleCliJsPaths) {
            try {
              cliContent = await Deno.readTextFile(possibleCliJs);
              break;
            } catch {
              // Try next path
            }
          }
          
          if (!cliContent) {
            // The claude binary might itself contain model references
            try {
              cliContent = await Deno.readTextFile(claudePath);
            } catch {
              // Give up on this path
            }
          }
        }
      } catch {
        // which/where failed
      }
    }

    if (!cliContent) {
      console.log("Model fetcher: Could not find Claude CLI binary");
      return null;
    }

    // Extract model IDs matching the pattern: claude-<family>-<version>-YYYYMMDD
    const modelPattern = /claude-[a-z0-9-]+-\d{8}/g;
    const matches = cliContent.match(modelPattern) || [];
    
    // Deduplicate and filter out non-model patterns
    const uniqueModels = [...new Set(matches)].filter(id => {
      // Must be a plausible model ID (contains a known family name)
      return id.includes('opus') || id.includes('sonnet') || id.includes('haiku') || id.includes('code');
    });

    if (uniqueModels.length === 0) {
      console.log("Model fetcher: No model IDs found in CLI binary");
      return null;
    }

    console.log(`Model fetcher: Discovered ${uniqueModels.length} models from CLI binary`);
    return uniqueModels;
  } catch (error) {
    console.error("Failed to parse models from CLI:", error instanceof Error ? error.message : String(error));
    return null;
  }
}

/**
 * Build models record from CLI-discovered model IDs.
 * Creates ModelInfo entries with inferred metadata.
 */
function buildModelsFromCLI(modelIds: string[]): Record<string, ModelInfo> {
  const models: Record<string, ModelInfo> = {};

  for (const id of modelIds) {
    const tier = classifyTier(id);
    const familyName = id.includes('opus') ? 'Opus' : id.includes('sonnet') ? 'Sonnet' : id.includes('haiku') ? 'Haiku' : 'Claude';
    
    // Extract version info from ID
    const versionMatch = id.match(/(\d+)(?:-(\d+))?-(\d{8})$/);
    const majorVersion = versionMatch ? versionMatch[1] : '';
    const minorVersion = versionMatch?.[2] ? `.${versionMatch[2]}` : '';
    
    models[id] = {
      name: `Claude ${familyName} ${majorVersion}${minorVersion}`,
      description: `Claude ${familyName} ${majorVersion}${minorVersion} (discovered from CLI)`,
      contextWindow: inferContextWindow(id),
      recommended: false,
      supportsThinking: inferSupportsThinking(id),
      tier,
      deprecated: inferDeprecated(id),
    };
  }

  // Build aliases from CLI-discovered models
  const apiLikeEntries: AnthropicModelEntry[] = modelIds.map(id => ({
    id,
    display_name: models[id]?.name || id,
    created_at: extractDateFromId(id),
    type: 'model',
  }));

  buildAliases(models, apiLikeEntries);

  return models;
}

/**
 * Extract a date string from a model ID (the YYYYMMDD suffix).
 */
function extractDateFromId(id: string): string {
  const match = id.match(/(\d{4})(\d{2})(\d{2})$/);
  if (match) {
    return `${match[1]}-${match[2]}-${match[3]}T00:00:00Z`;
  }
  return '2024-01-01T00:00:00Z';
}

/**
 * Fetch and cache models. Returns the models record.
 * Strategy: API key > CLI binary parsing > null (use hardcoded defaults).
 */
export async function fetchModels(): Promise<Record<string, ModelInfo> | null> {
  const now = Date.now();

  // Return cached if still fresh
  if (cachedModels && (now - lastFetchTime) < CACHE_TTL_MS) {
    return cachedModels;
  }

  // Strategy 1: Use the models API (custom base URL first, then Anthropic).
  // fetchFromAPI only returns a non-empty record, so this never caches {}.
  const apiModels = await fetchFromAPI();
  if (apiModels) {
    cachedModels = apiModels;
    lastFetchTime = now;
    return cachedModels;
  }

  // Strategy 2: Parse from installed CLI binary
  const cliModelIds = await parseModelsFromCLI();
  if (cliModelIds && cliModelIds.length > 0) {
    cachedModels = buildModelsFromCLI(cliModelIds);
    lastFetchTime = now;
    console.log(`Model fetcher: Loaded ${Object.keys(cachedModels).length} models from CLI binary`);
    return cachedModels;
  }

  // Fallback: use hardcoded defaults
  console.log("Model fetcher: No dynamic source available, using hardcoded defaults");
  return null;
}

/**
 * Start periodic model refresh (call once at startup).
 */
export function startModelRefresh(
  updateCallback: (models: Record<string, ModelInfo>) => void
): void {
  // Initial fetch
  fetchModels().then(models => {
    if (models) {
      updateCallback(models);
    }
  }).catch(err => {
    console.error("Initial model fetch failed:", err);
  });

  // Periodic refresh
  refreshInterval = setInterval(async () => {
    try {
      // Force re-fetch by clearing cache
      cachedModels = null;
      const models = await fetchModels();
      if (models) {
        updateCallback(models);
        console.log(`Model refresh: Updated ${Object.keys(models).length} models`);
      }
    } catch (err) {
      console.error("Model refresh failed:", err);
    }
  }, CACHE_TTL_MS);
}

/**
 * Stop periodic model refresh.
 */
export function stopModelRefresh(): void {
  if (refreshInterval) {
    clearInterval(refreshInterval);
    refreshInterval = null;
  }
}

/**
 * Clear the model cache (useful for testing or manual refresh).
 */
export function clearModelCache(): void {
  cachedModels = null;
  lastFetchTime = 0;
}
