const defaultQuery =
  "SComCP: Task-Oriented Semantic Communication for Collaborative Perception";

const app = document.querySelector(".app");
const homeForm = document.querySelector("#homeSearchForm");
const resultsForm = document.querySelector("#resultsSearchForm");
const homeQuery = document.querySelector("#homeQuery");
const resultsQuery = document.querySelector("#resultsQuery");
const backHome = document.querySelector("#backHome");
const refreshButton = document.querySelector("#refreshButton");
const pauseButton = document.querySelector("#pauseButton");
const downloadButton = document.querySelector("#downloadButton");
const shareButton = document.querySelector("#shareButton");
const timeTabs = Array.from(document.querySelectorAll(".time-tabs button"));
const strategyTabs = Array.from(document.querySelectorAll(".strategy-tabs button"));
const sourceLocal = document.querySelector("#sourceLocal");
const sourceOpenAlex = document.querySelector("#sourceOpenAlex");
const sourceArxiv = document.querySelector("#sourceArxiv");
const limitInput = document.querySelector("#limitInput");
const perQueryInput = document.querySelector("#perQueryInput");
const maxQueriesInput = document.querySelector("#maxQueriesInput");
const answerToggle = document.querySelector("#answerToggle");
const pipelinePlanner = document.querySelector("#pipelinePlanner");
const pipelineApis = document.querySelector("#pipelineApis");
const pipelineReranker = document.querySelector("#pipelineReranker");
const pipelineQwen = document.querySelector("#pipelineQwen");
const pipelineTiming = document.querySelector("#pipelineTiming");
const paperList = document.querySelector("#paperList");
const statusText = document.querySelector("#statusText");
const answerPanel = document.querySelector("#answerPanel");
const answerText = document.querySelector("#answerText");
const backResults = document.querySelector("#backResults");
const graphBackHome = document.querySelector("#graphBackHome");
const graphTitle = document.querySelector("#graphTitle");
const graphPaperInfo = document.querySelector("#graphPaperInfo");
const relationGraph = document.querySelector("#relationGraph");
const graphOriginItem = document.querySelector("#graphOriginItem");
const graphRelatedList = document.querySelector("#graphRelatedList");
const graphSearchInput = document.querySelector("#graphSearchInput");
const API_BASE =
  window.location.protocol === "file:"
    ? "http://localhost:5174"
    : window.location.origin;

const fallbackRows = [
  { title: "SComCP: Task-Oriented Semantic Communication for Collaborative Perception", year: 2024, source: "demo" },
  { title: "Task-Oriented Wireless Communications for Collaborative Perception in Intelligent Unmanned Systems", year: 2023, source: "demo" },
  { title: "CoDS: Collaborative Perception via Digital Semantic Communication", year: 2025, source: "demo" },
  { title: "Goal-Oriented Semantic Communication for Logical Decision Making", year: 2021, source: "demo" },
  { title: "R-ACP: Real-Time Adaptive Collaborative Perception Leveraging Robust Task-Oriented Communications", year: 2024, source: "demo" },
  { title: "Task-Oriented Semantic Communication in Large Multimodal Models-based Vehicle Networks", year: 2026, source: "demo" },
  { title: "Task-Oriented Semantic Communication for Stereo-Vision 3D Object Detection", year: 2022, source: "demo" },
  { title: "Semantic Communication for Cooperative Perception using HARQ", year: 2020, source: "demo" },
];

let currentPapers = fallbackRows;
let currentFilter = "any";
let currentStrategy = "standard";
let currentReranker = { enabled: false, type: "lexical" };
let currentTrace = null;
let currentTiming = {};
let selectedTitles = new Set();
let expandedTitles = new Set();
let graphRelatedItems = [];
let activeController = null;
let answerController = null;

function setView(view) {
  app.dataset.view = view;
  if (view === "home") {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
}

function readNumber(input, fallback, min, max) {
  const value = Number(input?.value);
  if (!Number.isFinite(value)) return fallback;
  return Math.max(min, Math.min(max, Math.round(value)));
}

function selectedSources() {
  const sources = [];
  if (sourceLocal?.checked) sources.push("local");
  if (sourceOpenAlex.checked) sources.push("openalex");
  if (sourceArxiv.checked) sources.push("arxiv");
  if (!sources.length) {
    if (sourceLocal) sourceLocal.checked = true;
    sourceOpenAlex.checked = true;
    sources.push("local");
    sources.push("openalex");
  }
  return sources;
}

function syncModeDefaults() {
  if (currentStrategy === "standard") {
    if (Number(maxQueriesInput.value) > 3) maxQueriesInput.value = 3;
    if (Number(perQueryInput.value) < 60) perQueryInput.value = 80;
    return;
  }
  if (Number(maxQueriesInput.value) < 5) maxQueriesInput.value = 5;
  if (Number(perQueryInput.value) < 40) perQueryInput.value = 50;
}

async function runSearch(query) {
  const normalized = query.trim() || defaultQuery;
  if (activeController) activeController.abort();
  if (answerController) answerController.abort();

  syncModeDefaults();
  const sources = selectedSources();
  const limit = readNumber(limitInput, 12, 5, 30);
  const perQuery = readNumber(perQueryInput, currentStrategy === "standard" ? 80 : 50, 5, 80);
  const maxQueries = readNumber(maxQueriesInput, currentStrategy === "standard" ? 3 : 5, 1, 8);

  activeController = new AbortController();
  answerController = null;
  pauseButton.classList.remove("active");
  homeQuery.value = normalized;
  resultsQuery.value = normalized;
  currentTiming = {};
  setPipeline({
    planner: currentStrategy === "standard" ? "Heuristic" : "Qwen",
    apis: sources.join(", "),
    reranker: "Waiting",
    qwen: answerToggle.checked ? "Waiting" : "Off",
    timing: "--",
  });
  statusText.textContent = "Searching local corpus and academic APIs...";
  renderAnswer(answerToggle.checked ? "Search is running. Qwen answer will be generated after papers arrive." : "Qwen answer generation is disabled.");
  renderRows([]);
  setView("results");

  try {
    const response = await fetch(`${API_BASE}/api/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: activeController.signal,
      body: JSON.stringify({
        query: normalized,
        planner: currentStrategy === "standard" ? "heuristic" : "qwen",
        answer: "none",
        sources: sources.join(","),
        max_queries: maxQueries,
        per_query: perQuery,
        limit,
        strategy: currentStrategy,
      }),
    });
    const payload = await response.json();
    if (!response.ok || payload.error) {
      throw new Error(payload.error || `Search failed with status ${response.status}`);
    }
    currentPapers = payload.papers || [];
    currentReranker = payload.reranker || { enabled: false, type: "lexical" };
    currentTrace = payload.pipeline_trace || null;
    currentTiming = payload.timing || {};
    selectedTitles = new Set();
    expandedTitles = new Set();
    renderRows(applyYearFilter(currentPapers));

    const planner = payload.query_plan?.planner || (currentStrategy === "standard" ? "heuristic" : "qwen");
    setPipeline({
      planner,
      apis: sources.join(", "),
      reranker: rerankerLabel(),
      qwen: answerToggle.checked ? "Generating" : "Off",
      timing: compactTiming(),
    });
    if (answerToggle.checked) {
      renderAnswer("Papers are ready. Qwen is generating the answer in the background...");
      statusText.textContent = `${pipelineSummary(planner)} ${timingSummary()}Generating Qwen answer...`;
      requestAnswer(normalized, currentPapers);
    } else {
      renderAnswer("Qwen answer generation is disabled. Retrieved papers are shown below.");
      statusText.textContent = `${pipelineSummary(planner)} ${timingSummary()}`;
    }
  } catch (error) {
    if (error.name === "AbortError") {
      statusText.textContent = "Search paused. Current rows are kept.";
      setPipeline({ qwen: "Paused" });
      return;
    }
    currentPapers = fallbackRows;
    currentReranker = { enabled: false, type: "demo" };
    currentTrace = null;
    renderRows(fallbackRows);
    renderAnswer("Search API unavailable, so no model-generated answer was produced.");
    statusText.textContent = `Search API unavailable: ${error.message}. Showing demo rows.`;
    setPipeline({ reranker: "Demo", qwen: "Unavailable", timing: "--" });
  } finally {
    activeController = null;
  }
}

async function requestAnswer(query, papers) {
  if (!papers.length) {
    renderAnswer("No papers were retrieved, so Qwen answer generation was skipped.");
    setPipeline({ qwen: "Skipped", timing: compactTiming() });
    return;
  }
  if (answerController) answerController.abort();
  answerController = new AbortController();
  renderAnswer("Qwen is synthesizing an evidence-based answer from the retrieved papers...");
  try {
    const response = await fetch(`${API_BASE}/api/answer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: answerController.signal,
      body: JSON.stringify({
        query,
        papers: papers.slice(0, 8),
      }),
    });
    const payload = await response.json();
    if (!response.ok || payload.error) {
      throw new Error(payload.error || `Answer failed with status ${response.status}`);
    }
    currentTiming = { ...currentTiming, ...(payload.timing || {}) };
    renderAnswer(withTimingFooter(payload.answer || "Qwen returned no answer."));
    statusText.textContent = `${pipelineSummary()} ${timingSummary()}Qwen answer is ready.`;
    setPipeline({ qwen: "Ready", timing: compactTiming() });
  } catch (error) {
    if (error.name === "AbortError") return;
    renderAnswer(`Qwen answer failed: ${error.message}`);
    setPipeline({ qwen: "Failed", timing: compactTiming() });
  } finally {
    answerController = null;
  }
}

function renderAnswer(answer) {
  answerPanel.hidden = false;
  answerText.textContent = answer;
}

function withTimingFooter(answer) {
  const summary = timingSummary();
  return summary ? `${answer}\n\n${summary}` : answer;
}

function renderRows(papers) {
  if (!papers.length) {
    paperList.innerHTML = `
      <article class="paper-row loading-row">
        <div class="paper-title">Searching local corpus and academic APIs...</div>
        <span class="paper-check" aria-hidden="true"></span>
        <button class="paper-remove" type="button" aria-label="Remove paper">x</button>
      </article>
    `;
    return;
  }

  paperList.innerHTML = papers
    .map((paper, index) => {
      const title = paper.title || "Untitled paper";
      const checked = selectedTitles.has(title) ? " checked" : "";
      const expanded = expandedTitles.has(title);
      const delay = Math.min(index * 70, 840);
      const meta = paper.year || paper.source || paper.score
        ? ` <span class="paper-meta-inline">${[paper.year, paper.source, scoreText(paper.score)].filter(Boolean).join(" | ")}</span>`
        : "";
      return `
        <article class="paper-row fade-in${expanded ? " expanded" : ""}" data-title="${escapeHtml(title)}" style="animation-delay: ${delay}ms">
          <button class="paper-main" type="button" aria-expanded="${expanded}" title="${escapeHtml(title)}">
            <span class="paper-title">${escapeHtml(title)}${meta}</span>
          </button>
          <button class="paper-check${checked}" type="button" aria-label="Select paper"></button>
          <button class="paper-remove" type="button" aria-label="Remove paper">x</button>
          ${expanded ? renderPaperDetails(paper) : ""}
        </article>
      `;
    })
    .join("");
}

function renderPaperDetails(paper) {
  const authors = Array.isArray(paper.authors) && paper.authors.length
    ? paper.authors.slice(0, 8).join(", ")
    : "Unknown authors";
  const abstract = paper.abstract || "No abstract is available for this result yet.";
  const reason = paper.reason || paper.relevance_reason || "";
  const url = paper.url || paper.doi || "";
  return `
    <div class="paper-detail">
      <div class="detail-grid">
        <div><span>Year</span><strong>${escapeHtml(String(paper.year || "Unknown"))}</strong></div>
        <div><span>Source</span><strong>${escapeHtml(paper.source || "Unknown")}</strong></div>
        <div><span>Citations</span><strong>${escapeHtml(String(paper.citation_count ?? 0))}</strong></div>
        <div><span>Score</span><strong>${scoreText(paper.score) || "n/a"}</strong></div>
      </div>
      <div class="reranker-badge">${escapeHtml(rerankerLabel())}</div>
      <p><strong>Authors:</strong> ${escapeHtml(authors)}</p>
      ${reason ? `<p><strong>Reranker reason:</strong> ${escapeHtml(reason)}</p>` : ""}
      <p><strong>Introduction:</strong> ${escapeHtml(abstract)}</p>
      <div class="detail-actions">
        <button class="relation-button" type="button" data-relation-title="${escapeHtml(paper.title || "")}">查看关联图</button>
        ${url ? `<a class="paper-link" href="${escapeHtml(url)}" target="_blank" rel="noreferrer">Open paper</a>` : ""}
      </div>
    </div>
  `;
}

function applyYearFilter(papers) {
  if (currentFilter === "any") return papers;
  const year = Number(currentFilter);
  if (!Number.isFinite(year)) return papers;
  return papers.filter((paper) => Number(paper.year) >= year);
}

function setYearFilter(value) {
  currentFilter = value;
  timeTabs.forEach((button) => {
    button.classList.toggle("active", button.dataset.yearFilter === value);
  });
  const filtered = applyYearFilter(currentPapers);
  renderRows(filtered);
  statusText.textContent = `${filtered.length} of ${currentPapers.length} papers shown for ${filterLabel(value)}.`;
}

function filterLabel(value) {
  return value === "any" ? "any time" : `since ${value}`;
}

function scoreText(score) {
  if (typeof score !== "number") return "";
  return `rel ${score.toFixed(2)}`;
}

function rerankerLabel() {
  if (currentReranker?.enabled) return currentReranker.type || "trainable reranker";
  return "lexical ranker";
}

function pipelineSummary(planner = "planner") {
  if (currentTrace?.strategy === "spar-one-layer") {
    return `${currentPapers.length} papers via SPAR: ${currentTrace.initial_candidates} initial, ${currentTrace.accepted_citations} citation, ${currentTrace.accepted_evolved} evolved; ${rerankerLabel()}.`;
  }
  const candidates = currentTrace?.initial_candidates ? `${currentTrace.initial_candidates} candidates, ` : "";
  return `${applyYearFilter(currentPapers).length} of ${currentPapers.length} papers shown via ${planner} + ${rerankerLabel()} (${candidates}${selectedSources().join(", ")}).`;
}

function timingSummary() {
  const parts = [];
  if (typeof currentTiming.search_seconds === "number") {
    parts.push(`文献搜索耗时 ${currentTiming.search_seconds.toFixed(2)}s`);
  }
  if (typeof currentTiming.llm_answer_seconds === "number" && currentTiming.llm_answer_seconds > 0) {
    parts.push(`大语言模型耗时 ${currentTiming.llm_answer_seconds.toFixed(2)}s`);
  }
  return parts.length ? `${parts.join("，")}。` : "";
}

function compactTiming() {
  const parts = [];
  if (typeof currentTiming.search_seconds === "number") {
    parts.push(`Search ${currentTiming.search_seconds.toFixed(1)}s`);
  }
  if (typeof currentTiming.llm_answer_seconds === "number" && currentTiming.llm_answer_seconds > 0) {
    parts.push(`LLM ${currentTiming.llm_answer_seconds.toFixed(1)}s`);
  }
  return parts.length ? parts.join(" / ") : "--";
}

function setPipeline(next) {
  if (next.planner !== undefined) pipelinePlanner.textContent = next.planner || "--";
  if (next.apis !== undefined) pipelineApis.textContent = next.apis || "--";
  if (next.reranker !== undefined) pipelineReranker.textContent = next.reranker || "--";
  if (next.qwen !== undefined) pipelineQwen.textContent = next.qwen || "--";
  if (next.timing !== undefined) pipelineTiming.textContent = next.timing || "--";
}

function openRelationGraph(title) {
  const focus = currentPapers.find((paper) => paper.title === title);
  if (!focus) {
    statusText.textContent = "Cannot build relation graph for this paper.";
    return;
  }
  renderRelationGraph(focus);
  setView("graph");
}

function renderRelationGraph(focus) {
  const related = buildRelatedPapers(focus, currentPapers).slice(0, 22);
  graphRelatedItems = related;
  const nodes = [focus, ...related.map((item) => item.paper)];
  const width = 980;
  const height = 680;
  const center = { x: width * 0.48, y: height * 0.55 };

  graphTitle.textContent = focus.title || "Paper relation graph";
  graphOriginItem.innerHTML = renderGraphListItem(focus, null, true);
  renderGraphList(related);
  graphPaperInfo.innerHTML = `
    <strong>${escapeHtml(focus.title || "Untitled paper")}</strong>
    <span>${escapeHtml([focus.year, focus.source, scoreText(focus.score)].filter(Boolean).join(" | ") || "No metadata")}</span>
    <p>${escapeHtml((focus.abstract || "No abstract is available.").slice(0, 520))}</p>
  `;

  const positions = nodes.map((paper, index) => {
    if (index === 0) return { paper, x: center.x, y: center.y, relation: null, size: 62 };
    const relation = related[index - 1];
    const ring = index <= 9 ? 205 : 285;
    const angle = ((index - 1) * 2.399963) - Math.PI / 2;
    const jitterX = Math.sin(index * 1.73) * 34;
    const jitterY = Math.cos(index * 2.11) * 28;
    const size = 26 + Math.min(36, Math.max(0, relation.score) * 55);
    return {
      paper,
      x: Math.max(70, Math.min(width - 70, center.x + Math.cos(angle) * ring + jitterX)),
      y: Math.max(70, Math.min(height - 70, center.y + Math.sin(angle) * ring + jitterY)),
      relation,
      size,
    };
  });

  const centerEdges = positions
    .slice(1)
    .map((node) => {
      const label = node.relation?.label || "related";
      const strong = (node.relation?.score || 0) >= 0.35 ? " strong" : "";
      const lx = (center.x + node.x) / 2;
      const ly = (center.y + node.y) / 2;
      return `
        <line class="graph-edge${strong}" x1="${center.x}" y1="${center.y}" x2="${node.x}" y2="${node.y}"></line>
        <text class="edge-label" x="${lx}" y="${ly}">${escapeHtml(label)}</text>
      `;
    })
    .join("");

  const crossEdges = buildCrossEdges(positions.slice(1))
    .map(([a, b]) => `<line class="graph-edge cross" x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}"></line>`)
    .join("");

  const renderedNodes = positions
    .map((node, index) => {
      const isCenter = index === 0;
      const title = trimTitle(node.paper.title || "Untitled paper", isCenter ? 34 : 28);
      const meta = [node.paper.year, node.paper.source].filter(Boolean).join(" | ");
      return `
        <g class="graph-node${isCenter ? " center" : ""}" transform="translate(${node.x}, ${node.y})">
          <circle r="${node.size}"></circle>
          <text text-anchor="middle" y="-4">${escapeHtml(title)}</text>
          <text class="node-meta" text-anchor="middle" y="15">${escapeHtml(meta || "related paper")}</text>
        </g>
      `;
    })
    .join("");

  relationGraph.setAttribute("viewBox", `0 0 ${width} ${height}`);
  relationGraph.innerHTML = crossEdges + centerEdges + renderedNodes;
}

function renderGraphList(items) {
  const query = normalizeToken(graphSearchInput.value);
  const filtered = query
    ? items.filter((item) => normalizeToken(item.paper.title).includes(query) || normalizeToken((item.paper.authors || []).join(" ")).includes(query))
    : items;
  graphRelatedList.innerHTML = filtered
    .map((item) => renderGraphListItem(item.paper, item, false))
    .join("");
}

function renderGraphListItem(paper, relation, origin) {
  const authors = Array.isArray(paper.authors) && paper.authors.length
    ? paper.authors.slice(0, 3).join(", ")
    : "Unknown authors";
  const label = origin ? "Origin paper" : relation?.label || "Related paper";
  return `
    <article class="graph-list-item${origin ? " origin" : ""}">
      <div class="graph-list-label">${escapeHtml(label)}</div>
      <strong>${escapeHtml(paper.title || "Untitled paper")}</strong>
      <p>${escapeHtml(authors)}</p>
      <span>${escapeHtml([paper.year, paper.source].filter(Boolean).join(" | ") || "No metadata")}</span>
    </article>
  `;
}

function buildCrossEdges(nodes) {
  const edges = [];
  for (let i = 0; i < nodes.length; i += 1) {
    for (let j = i + 1; j < nodes.length; j += 1) {
      const leftTerms = keywordSet(`${nodes[i].paper.title || ""} ${nodes[i].paper.abstract || ""}`);
      const rightTerms = keywordSet(`${nodes[j].paper.title || ""} ${nodes[j].paper.abstract || ""}`);
      const overlap = intersectionSize(leftTerms, rightTerms);
      if (overlap >= 4 && edges.length < 38) {
        edges.push([nodes[i], nodes[j]]);
      }
    }
  }
  return edges;
}

function buildRelatedPapers(focus, papers) {
  const focusAuthors = new Set((focus.authors || []).map(normalizeToken).filter(Boolean));
  const focusTerms = keywordSet(`${focus.title || ""} ${focus.abstract || ""}`);
  return papers
    .filter((paper) => paper.title && paper.title !== focus.title)
    .map((paper) => {
      const authors = new Set((paper.authors || []).map(normalizeToken).filter(Boolean));
      const terms = keywordSet(`${paper.title || ""} ${paper.abstract || ""}`);
      const authorOverlap = intersectionSize(focusAuthors, authors);
      const topicOverlap = intersectionSize(focusTerms, terms);
      const sameSource = focus.source && paper.source && focus.source === paper.source ? 1 : 0;
      const yearDistance = Number.isFinite(Number(focus.year)) && Number.isFinite(Number(paper.year))
        ? Math.abs(Number(focus.year) - Number(paper.year))
        : 10;
      const yearScore = Math.max(0, 1 - yearDistance / 8);
      const score =
        authorOverlap * 0.28 +
        Math.min(topicOverlap, 10) * 0.055 +
        sameSource * 0.08 +
        yearScore * 0.05 +
        (typeof paper.score === "number" ? paper.score * 0.08 : 0);
      const label = authorOverlap > 0
        ? "shared authors"
        : topicOverlap >= 3
          ? "topic overlap"
          : "nearby result";
      return { paper, score, label };
    })
    .sort((a, b) => b.score - a.score);
}

function keywordSet(text) {
  const stopwords = new Set([
    "the", "and", "for", "with", "from", "into", "that", "this", "using", "based",
    "paper", "papers", "study", "survey", "related", "about", "are", "what", "which",
    "相关", "论文", "研究", "基于", "关于", "哪些", "推荐", "代表性",
  ]);
  return new Set(
    String(text)
      .toLowerCase()
      .replace(/[^\p{L}\p{N}\s]+/gu, " ")
      .split(/\s+/)
      .map((token) => token.trim())
      .filter((token) => token.length >= 2 && !stopwords.has(token))
  );
}

function normalizeToken(value) {
  return String(value || "").trim().toLowerCase();
}

function intersectionSize(left, right) {
  let count = 0;
  left.forEach((item) => {
    if (right.has(item)) count += 1;
  });
  return count;
}

function trimTitle(title, maxLength) {
  const clean = String(title).replace(/\s+/g, " ").trim();
  return clean.length > maxLength ? `${clean.slice(0, maxLength - 1)}...` : clean;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

homeForm.addEventListener("submit", (event) => {
  event.preventDefault();
  runSearch(homeQuery.value);
});

resultsForm.addEventListener("submit", (event) => {
  event.preventDefault();
  runSearch(resultsQuery.value);
});

backHome.addEventListener("click", () => {
  setView("home");
});

backResults.addEventListener("click", () => {
  setView("results");
});

graphBackHome.addEventListener("click", () => {
  setView("home");
});

graphSearchInput.addEventListener("input", () => {
  renderGraphList(graphRelatedItems);
});

refreshButton.addEventListener("click", () => {
  runSearch(resultsQuery.value);
});

pauseButton.addEventListener("click", () => {
  if (activeController) {
    activeController.abort();
    pauseButton.classList.add("active");
  }
  if (answerController) {
    answerController.abort();
    pauseButton.classList.add("active");
  }
});

downloadButton.addEventListener("click", () => {
  const payload = {
    query: resultsQuery.value,
    filter: currentFilter,
    selected_titles: Array.from(selectedTitles),
    papers: applyYearFilter(currentPapers),
    timing: currentTiming,
    pipeline: currentTrace,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "scholarcp-results.json";
  link.click();
  URL.revokeObjectURL(url);
});

shareButton.addEventListener("click", async () => {
  const text = `${resultsQuery.value}\n${applyYearFilter(currentPapers)
    .slice(0, 8)
    .map((paper, index) => `${index + 1}. ${paper.title}`)
    .join("\n")}`;
  try {
    await navigator.clipboard.writeText(text);
    statusText.textContent = "Result summary copied to clipboard.";
  } catch {
    statusText.textContent = "Clipboard unavailable. Use Download instead.";
  }
});

timeTabs.forEach((button) => {
  button.addEventListener("click", () => {
    const value = button.dataset.yearFilter;
    if (value === "custom") {
      const input = window.prompt("Show papers since year:", "2024");
      if (!input) return;
      const year = Number(input);
      if (!Number.isInteger(year) || year < 1900 || year > 2100) {
        statusText.textContent = "Invalid custom year.";
        return;
      }
      button.textContent = `Since ${year}`;
      button.dataset.yearFilter = String(year);
      setYearFilter(String(year));
      return;
    }
    setYearFilter(value);
  });
});

strategyTabs.forEach((button) => {
  button.addEventListener("click", () => {
    currentStrategy = button.dataset.strategy || "standard";
    strategyTabs.forEach((tab) => tab.classList.toggle("active", tab === button));
    syncModeDefaults();
    runSearch(resultsQuery.value || homeQuery.value);
  });
});

[sourceLocal, sourceOpenAlex, sourceArxiv, limitInput, perQueryInput, maxQueriesInput, answerToggle].forEach((control) => {
  if (!control) return;
  control.addEventListener("change", () => {
    setPipeline({ apis: selectedSources().join(", ") });
  });
});

paperList.addEventListener("click", (event) => {
  const relationButton = event.target.closest(".relation-button");
  if (relationButton) {
    openRelationGraph(relationButton.dataset.relationTitle || "");
    return;
  }

  const checkButton = event.target.closest(".paper-check");
  if (checkButton) {
    const row = checkButton.closest(".paper-row");
    const title = row?.dataset.title || "";
    if (selectedTitles.has(title)) {
      selectedTitles.delete(title);
      checkButton.classList.remove("checked");
    } else {
      selectedTitles.add(title);
      checkButton.classList.add("checked");
    }
    statusText.textContent = `${selectedTitles.size} papers selected.`;
    return;
  }

  const removeButton = event.target.closest(".paper-remove");
  if (removeButton) {
    const row = removeButton.closest(".paper-row");
    const title = row?.dataset.title || "";
    currentPapers = currentPapers.filter((paper) => paper.title !== title);
    selectedTitles.delete(title);
    expandedTitles.delete(title);
    renderRows(applyYearFilter(currentPapers));
    statusText.textContent = `${applyYearFilter(currentPapers).length} papers kept after manual filtering.`;
    return;
  }

  const mainButton = event.target.closest(".paper-main");
  if (mainButton) {
    const row = mainButton.closest(".paper-row");
    const title = row?.dataset.title || "";
    if (expandedTitles.has(title)) {
      expandedTitles.delete(title);
    } else {
      expandedTitles.add(title);
    }
    renderRows(applyYearFilter(currentPapers));
  }
});

homeQuery.value = "";
resultsQuery.value = defaultQuery;
renderRows(fallbackRows);
renderAnswer("The model-generated answer will appear here after retrieval.");
setPipeline({ planner: "Ready", apis: "local, openalex", reranker: "Lexical", qwen: "Idle", timing: "--" });

