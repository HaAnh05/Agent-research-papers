# Research Scout — UI Design Specification

Status: final visual and interaction contract (v1). This document specifies the frontend; it does not change the LangGraph workflow. Where an event or artifact is not exposed by the current backend, the UI must omit that detail until an adapter can provide it truthfully.

## 1. Product thesis and scope

Research Scout is a quiet research instrument for someone who wants to turn a paper, PDF, arXiv reference, or research question into readable, inspectable findings. Its single initial job is to accept that research request. Its signature interaction is a **semantic account of work in progress**: the user sees meaningful research stages while the internal graph remains in the background.

The visual sequence is **Manus-like minimal landing → semantic LangGraph pipeline → Research-Copilot-like result workspace**. These are successive application states, not three dashboard panels shown together. Beautiful UI contributes implementation primitives and interaction behavior, **not** the product's visual identity. Do not reproduce another product's branding or interface wholesale.

The backend is **one stateful LangGraph research workflow**, not a swarm. It has conditional search/refinement, direct-input routing, PDF reading, GitHub/BibTeX enrichment, PMRL notes, optional multi-paper comparison, final Markdown report, append-only trace/error logs, and in-memory checkpoints. Do not invent chat agents, durable jobs, a database, vector search, or a human-approval step.

### Non-negotiable sequence

```text
Landing --submit--> Pipeline --successful completion--> Results
                         |--recoverable/fatal failure--> Pipeline error state
```

While running, the pipeline may show verified counts and activity. **Do not render PMRL cards, benchmark, report, or speculative “early synthesis” before the corresponding output is finalized.** A prior exploratory sketch mentioned progressive synthesis; the later three-screen decision supersedes it. On completion, replace the pipeline view with the result workspace rather than stacking them. A compact “Activity” control may remain in Results for provenance.

## 2. Visual tokens — use these, not framework defaults

Load Newsreader, Inter, and JetBrains Mono with appropriate font-display behavior. If a font cannot load, use the listed fallback; do not substitute another design face. Newsreader occurs **only on the Landing headline**. Inter carries all result content and controls. JetBrains Mono is confined to trace timestamps, IDs, filenames, and technical metadata.

```css
:root {
  --font-display: "Newsreader", Georgia, serif;
  --font-ui: "Inter", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, SFMono-Regular, monospace;

  --background: #FAFAF8;
  --surface: #FFFFFF;
  --surface-subtle: #F5F5F1;
  --surface-hover: #F1F1ED;
  --text-primary: #181817;
  --text-secondary: #5F5F59;
  --text-muted: #92928B;
  --border: #E5E5DF;
  --border-strong: #D6D6CF;
  --selected: #191918;
  --selected-text: #FAFAF8;
  --accent: #587FD3;
  --accent-subtle: #EEF3FC;
  --accent-ink: #315DAE; /* small link text; preserve contrast */
  --success: #609274;
  --error: #B75C58;

  --shadow-input: 0 8px 30px rgba(0, 0, 0, .028);
  --shadow-floating: 0 12px 36px rgba(0, 0, 0, .045);

  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-6: 24px;
  --space-8: 32px;
  --space-10: 40px;
  --space-12: 48px;
  --space-16: 64px;
  --space-20: 80px;

  --radius-badge: 4px;
  --radius-action: 6px;
  --radius-chip: 8px;
  --radius-source: 10px;
  --radius-card: 12px;
  --radius-pipeline: 14px;
  --radius-prompt: 18px;
  --radius-pill: 999px;
}
```

| Role | Font / exact metrics | Use |
| --- | --- | --- |
| Landing headline | Newsreader 400; `clamp(48px, 5vw, 64px)` / 1.02 / `-.035em` | One sentence only |
| Paper title | Inter 600; 28px / 1.16 / `-.028em` | Selected paper header |
| Page title | Inter 600; 20px / 1.25 / `-.02em` | Results/report heading |
| Section title | Inter 600; 14px / 1.4 | PMRL, TL;DR, Activity |
| Reading body | Inter 400; 14px / 1.65 | Paragraphs; may increase to 15px for long-form report |
| Control / UI label | Inter 500; 12px / 1.4 | Tabs, buttons, stage labels |
| Metadata | Inter 400 or 500; 11px / 1.45 | Supporting dates and counts only |
| Trace | JetBrains Mono 400; 11px / 1.5 | Activity details only |

Use weights 400/500/600; no blanket bold text. Reading text and essential state labels must use `--text-primary` or `--text-secondary`, not `--text-muted`. The muted color is for nonessential metadata or decorative dots. Links use `--accent-ink`; the softer `--accent` is for active dots, relevance accents, and focus treatment. At least ~90% of the visual field remains warm neutral. Status must be communicated by words and shape, never color alone.

Default line border: `1px solid var(--border)`; TL;DR border: `1px solid #DDDDD6`. PMRL content surfaces have **no shadow**. Only the Landing composer uses `--shadow-input` and the Pipeline panel uses `--shadow-floating`. Never combine a strong shadow, prominent border, and tinted fill on the same surface.

## 3. Screen A — Landing

No navbar, top progress rail, sidebar, history, recent papers, or settings in the initial viewport. Give the user one obvious action. Content max-width **840px**, composer width `min(100%, 820px)`, minimum height **132px**, radius **18px**. Desktop horizontal padding **40px** (32px at narrower desktop). Position the hero cluster around **30–35% from the viewport top**, not mathematically centered. Maintain 16px headline-to-subtitle, 32px subtitle-to-composer, and 16px composer-to-suggestions.

```text
                  Research Scout

       What do you want to understand?
          Give me a paper, topic or PDF.

       ┌─────────────────────────────────┐
       │ Ask about a paper or topic…    │
       │                                 │
       │ +                            ↑  │
       └─────────────────────────────────┘

        Analyze paper   Scout literature   Compare papers
```

The small product identity is Inter, not another large brand hero. The three suggestion pills are optional input affordances, not navigation or separate agents. Selecting one may prefill or focus the composer; it must not silently launch a task. Keep the composer to textarea, attachment, and submit. No model selector, provider branding, voice control, slash-command menu, source picker, or multiple competing calls to action.

Accept topic text, arXiv ID/URL, supported PDF URL, or attached PDF according to actual backend validation. A browser cannot directly read an arbitrary local filesystem path; attachment must upload through the application's supported API. Show the filename and an accessible remove action before submission. Disable submit for an empty request, while upload/validation is unresolved, or when required input is invalid; show a specific inline reason. Enter submits unless Shift+Enter inserts a newline. Do not pretend an unsupported file type was accepted.

## 4. Transition and Screen B — Pipeline

On submit, prevent duplicate runs, retain the request, and transition without a blank route flash. For 100–150ms the composer holds position; fade the Landing heading; scale the composer from 1 to .985; fade the Landing over **180ms ease-out**; reveal the Pipeline panel near the same visual center. Movement is at most 4–6px. On reduced-motion preference, replace transforms with an immediate state change or a minimal opacity change.

Pipeline width **560–620px** (`min(100%, 620px)`), white surface, 20–24px padding, 14px radius. Its heading says “Running research”; one sentence names the current semantic action, and a second line may give a verified count. Show elapsed time only while genuinely running, not a fake percentage or time estimate. There is no large progress bar, terminal, node graph, or full-screen spinner.

```text
Running research                                      13.4s
Finding and ranking papers…

  ● Understand request                         Done
  ● Discover relevant papers                   Running
  ○ Read source PDFs
  ○ Enrich sources
  ○ Write structured notes
  ○ Compare selected papers                    [if applicable]
  ○ Generate final report

  searched · 5 papers     selected · 3 papers
  Activity · 7 events                           [expand]
```

Stage row height at least **40px**; gap **4px**; active row may use `--surface-subtle` with 8px internal padding and no added border. Do not create one row per tool call or query refinement. The live headline and counts come from actual emitted state, not a simulated timer. A skipped conditional stage is removed from the compact list after routing, but remains recorded as “Skipped” in Activity when applicable.

### Stage model and graph mapping

The table defines **semantic UI IDs**, not assumed Python function names or literal wire event strings. The frontend adapter must inspect the existing `graph.py`, `state.py`, and trace payloads to map real node starts/ends/state updates into these IDs.

| UI stage ID and label | Workflow source | Running copy / completion evidence |
| --- | --- | --- |
| `understand` — Understand request | Router | “Understanding your request…”; complete when route/input classification is known |
| `discover` — Discover relevant papers | ArXiv Search + relevance Evaluate ↔ Refine | “Finding and ranking papers…”; count candidates/selected only from actual state; refinement increments an attempt detail, not a new row |
| `read` — Read source PDFs | Concurrent PDF reader | “Reading source PDFs…”; paper/page counts only when measured; for direct input label “Read supplied source(s)” |
| `enrich` — Enrich sources | GitHub lookup + BibTeX generation | “Checking related repositories and citations…”; reflect independent success/absence for each enrichment |
| `notes` — Write structured notes | PMRL writer | “Writing structured notes…”; complete only for finalized notes, not merely an LLM call start |
| `compare` — Compare selected papers | Optional multi-paper benchmark | “Comparing selected papers…”; present only if comparison is actually requested/planned and run |
| `report` — Generate final report | Final Markdown report generator | “Preparing the research report…”; completion requires a valid final artifact and graph completion |
| error annotation | Error handler / failure trace | Attach failure to current stage; never create a fictitious eighth research step |

The search implementation currently retrieves up to 5 candidates, selects up to 3, and refines at most twice. These are current workflow bounds, **not** a hard-coded UI promise. One `discover` row can show “Refined query · 1 time” or “2 times” in its detail; attempts remain in Activity. The reader may process up to four PDFs concurrently, inspect the first eight pages and retain limited extracted text. Do not imply a full-document inspection or precise page evidence when the reader has not supplied it.

#### Routing cases

| Case | Visible stage sequence | Requirement |
| --- | --- | --- |
| Topic/query | Understand → Discover → Read → Enrich → Notes → optional Compare → Report | Show search/evaluate/refine inside Discover; no duplicate rows on retry |
| Direct arXiv ID/URL or PDF | Understand → Read supplied source(s) → Enrich → Notes → optional Compare → Report | Do **not** animate or claim an arXiv search that did not run; record search as skipped only in Activity |
| Single paper, no comparison | Understand → route-specific stages → Notes → Report | No Comparison row or tab; Activity may state “Comparison skipped · single-paper task” if reported |
| Multiple papers without a produced benchmark | Same as above | Paper count alone does not justify an invented comparison matrix |
| Search with no suitable result | Stay in Pipeline error/empty state | Explain no papers met the selection criteria; offer an edit/retry path supported by the app |

If routing is not yet known, show only “Understand request” as active and do not guess future branch-specific work. The rest of the stage list becomes stable after routing. A refinement or node retry updates the existing row and attempt count; completed stages do not oscillate backward due to duplicate/out-of-order events.

### UI event adapter contract

Normalize actual graph events/traces and state snapshots into one typed UI model. Do not couple view components to raw Python node strings. Recommended shape (rename fields to match the real API):

```ts
type StageId = "understand" | "discover" | "read" | "enrich" | "notes" | "compare" | "report";
type StageStatus = "pending" | "running" | "completed" | "skipped" | "failed";
type RunStatus = "starting" | "running" | "failed" | "completed";

type UiEvent = {
  runId: string;
  sequence: number;          // monotonic, for ordering and de-duplication
  timestamp: string;
  stage?: StageId;
  kind: "stage_started" | "stage_completed" | "stage_skipped" |
        "stage_failed" | "stage_retried" | "count_updated" |
        "artifact_ready" | "run_completed" | "run_failed";
  publicMessage?: string;    // safe, non-sensitive UI copy
  counts?: { candidates?: number; selected?: number; pdfsRead?: number;
             repositories?: number; bibtexEntries?: number };
  artifact?: "pmrl" | "comparison" | "report";
};
```

This is a frontend normalization target, **not a claim that these named events already exist**. An SSE stream is appropriate if the backend exposes one; otherwise poll run state/trace snapshots at a measured interval. If neither live channel exists, show an honest indeterminate running state and transition only when the completed response arrives. Use stable `runId` and sequence/trace identity where available; don't manufacture graph progress. In-memory checkpoints are not durable: after reload, offer reconnection only if a real run endpoint supports it; otherwise explain that the run cannot be resumed and allow starting again.

### Row states and failure handling

| Status | Treatment |
| --- | --- |
| `pending` | `#D4D4CE` small dot, secondary text |
| `running` | blue dot, restrained pulse, primary label, optional subtle fill |
| `completed` | small muted-green dot, “Done” accessible text |
| `skipped` | muted dot and “Skipped” in Activity; hidden from compact list after route resolves |
| `failed` | muted-red dot, plain-language failure, an available recovery action |

Only the running dot may pulse. No giant green checkmarks, spinner on every row, fake percentage, or indefinite shimmer. A retry is a detail on the **same** row. If a real stage-level retry exists, label its action “Retry [stage]”; if it does not, show “Start again” and initiate a fresh run. Never promise continuation from an in-memory checkpoint after process restart. Retain the original request so users can edit it. A provider/configuration failure in `write_notes` needs a real visible failure path even if it currently escapes the intended backend handler; do not leave the UI permanently “Writing…” when the request terminates abnormally. For partial upstream enrichment failure, use the backend's actual graceful-degradation result: state what is missing, continue only if the workflow did, and never fabricate GitHub/BibTeX artifacts.

Error copy names the failed action, a safe cause when known, and the next available action. Example: “Could not write structured notes. The language-model provider did not respond. Start again.” Keep stack traces and sensitive diagnostics in a protected developer log, not the public view. If the connection is lost, distinguish “Connection lost; run status unknown” from “Research failed.”

## 5. Activity and trace transparency

Below the rows, render compact factual chips such as `searched · 5 papers`, `selected · 3 papers`, `read · 3 PDFs`, `GitHub · 2 related repos`, `BibTeX · 3 entries`. Only render a chip when the value exists; a missing value is not zero. The expandable **Activity** control replaces Beautiful UI's “Thinking” wording. It is collapsed by default and lists timestamped, append-only lifecycle events, refinement/retry counts, skipped stages, and safe errors in chronological order. Keep a stable scroll position while new entries arrive and provide an accessible live announcement for material stage changes, not every trace line.

Activity is **execution provenance**, never hidden chain-of-thought or a raw model-reasoning transcript. Sanitize credentials, local private paths, prompt internals, and oversized payloads. If an event has no public-safe detail, show its semantic stage change rather than a blank or confidential dump. The result workspace may expose this history through a quiet “Activity” disclosure; it must not compete with findings.

## 6. Screen C — Result workspace

Enter only after the run successfully finishes and the finalized outputs are available. Fade the Pipeline panel over **150–180ms** and reveal the workspace; do not render pipeline and full report simultaneously. Whole application max-width **1400px**, centered. Desktop source rail **248px** on `--background` with right border; the remaining content is fluid, with a maximum reading width **850px**, padding **36px 44px**. Let unused wide-screen space remain empty instead of stretching paragraphs.

```text
┌──────────────────────┬───────────────────────────────────────────┐
│ Sources              │ Summary   [Comparison]   Report           │
│ #01 title            │                                           │
│ #02 selected         │ Paper 2 of 3                              │
│ #03 title            │ Title · authors · date · categories       │
│                      │ arXiv  PDF  Related repository  BibTeX    │
│                      │                                           │
│                      │ TL;DR (full width)                       │
│                      │ Problem          Method                  │
│                      │ Key Results      Why it matters          │
│                      │ [View details: full PMRL]                │
│                      │ Evidence [expand if sourced]              │
└──────────────────────┴───────────────────────────────────────────┘
```

“Summary” is the default tab. Show “Comparison” **only when a real benchmark artifact was produced**; show “Report” only when a finalized Markdown report exists. For a completed single-paper task, this normally means Summary and Report. Tab change is an in-workspace view change, not a separate multi-step wizard. Preserve selected paper when moving between tabs. No top “01 Ask — 02 Research — 03 Result” bar.

### Source rail

Each source item shows rank (`#01`), title (two or three line clamp), and relevance **only if a real score exists**. Never display a fabricated “100%”. Selected item: `--selected` background, `--selected-text`, 10px radius, no glow. Unselected items are transparent; hover uses `--surface-hover`. Use a narrow blue relevance rule only when score is actually meaningful. No stack of white shadowed paper cards. A source remains selectable by keyboard; expose the selected state semantically.

### Summary / PMRL detail

Selected paper header order: “Paper n of N” → official arXiv title → authors, submission date, versioned arXiv ID, subject chips → compact source actions. If official metadata cannot be verified, use the ID as the title fallback and say which metadata is missing. The default Summary has exactly five distinct cards: a full-width TL;DR (one or two sentences) and a two-column grid of Problem, Method, Key Results, and Why it matters. The four smaller cards contain 30–60 generated words each; the whole summary is about 180–300 words. Invalid or legacy summaries show a missing state, never sliced PMRL prose. The five cards are generated after report synthesis inside the existing `final_report` node; the LangGraph nodes and edges remain unchanged.

Keep the four native PMRL fields behind a **View details** disclosure. When opened, render Problem, Method, Result, and Limitation in one readable column with the shared Markdown/math renderer. Use 12px radius, 1px border, 20–24px padding, and no shadow. If a field is unavailable, say “Not extracted” with a reason when known; do not silently fill it from a guess. Important formulas use separate LaTeX display blocks with short explanations and local horizontal scrolling. The PDF/HTML extraction may still miss tables or equations, so display measured coverage and distinguish an incomplete extraction from a paper that truly lacks experiments. Keep “Why it matters” as a grounded implication, distinct from measured results.

### GitHub, BibTeX, source actions

Keep arXiv and PDF links, any repository link, and BibTeX action together beneath the paper metadata; do not turn them into large feature cards. Repository matching is heuristic: label it **“Related repository”** or **“Likely repository”**, never “Official repository” unless independently verified and represented by backend data. Hide a missing link or use a quiet “No related repository found” in details, not a dead button. BibTeX action copies/shows the actual generated entry; if generation failed, mark it unavailable. External links open safely with appropriate `rel` attributes, show the destination in accessible text, and never invent URLs.

### Comparison and final report

Comparison is a restrained structured artifact with one short findings row per paper and an optional source-backed metric matrix. Put two values beside each other only when the artifact confirms the same metric, dataset, unit, and supporting source quote for both papers; otherwise label the metric “Not directly comparable”. Use a table on desktop and stack each paper as a readable card row on mobile. The saved benchmark Markdown remains available in a disclosure for provenance. A legacy run that has only Markdown shows that source without reconstructing a duplicate PMRL grid. No filters, sortable CRM grid, pagination, chart, or confidence meter. Keep the comparison container bounded so any table overflow stays inside it and the page itself never scrolls horizontally. If there are multiple papers but no benchmark output, omit the tab.

Render the final Markdown report as a readable article, preserving its actual headings, paragraphs, lists, tables, links, and citations. Show the first report heading once in the on-screen report header; hide only that visual duplicate from the article renderer while keeping the original Markdown unchanged for download. Put the technical filename in the supporting Download metadata. Do not print raw Markdown syntax or impose fabricated sections (“Research gaps”, etc.) that the report does not contain. Render Markdown safely: a verified HTTPS link is clickable, while an unverified placeholder remains text with a visible warning. If a real report file/download endpoint exists, show its filename and a Download action; copying the rendered report is optional if supported. No fake download for a backend-local path inaccessible to the browser.

The Activity disclosure groups duplicate completion notices from the same node while retaining retries and errors as separate entries. Completed nodes may show their measured duration. Source quality metadata appears beside the selected source when heading extraction or PDF parsing used a fallback or degraded path; unknown quality stays explicitly unknown.

## 7. Beautiful UI adoption map

The [Beautiful UI component gallery](https://www.beautifului.dev/) provides the named primitives. Copy/adapt implementation only after checking the project's frontend stack and the [license](https://www.beautifului.dev/license); preserve notices where required. The Python LangGraph workflow remains intact. A React/TypeScript frontend may consume actual backend data via HTTP/SSE; this document does not require a framework migration solely for visual fidelity.

| Research Scout component | Beautiful UI starting point | Adaptation boundary |
| --- | --- | --- |
| `ResearchPrompt` | Prompt Bar | Keep textarea, attachment, submit; remove model/voice/commands |
| `PipelineStatus` | Loading State | Keep restrained activity indicator and elapsed time; remove prominent pixel shimmer |
| `StageList` | Task Rows | Map semantic research stages, five row statuses, retry details |
| `ActivityChips` | Tool Chips | Verified compact counts, secondary placement |
| `EvidenceDisclosure` | Context Cards | Only genuine source excerpts/citations; collapsed by default |

Optional: adapt the mechanics of “Thinking” **only** for the renamed Activity disclosure, never for model reasoning. PMRL presentation is custom to Research Scout; “Insight Cards” may inform interaction mechanics but not charts, pagination, or styling. Do not use Approval Card while HITL is unimplemented. Do not import the gallery's Chat, Records Table, Filter Table, Sidebar Nav, Agent Screen, or Flowchart merely because they exist. Use one consistent thin-stroke icon set (Lucide or Iconoir, 1.5–1.75px), with icons only for actual actions/status. Avoid an icon-library dependency introduced solely by a component not used here.

## 8. Responsive and accessible behavior

| Viewport | Layout contract |
| --- | --- |
| ≥1200px | 248px source rail + fluid content; article max 850px; page max 1400px |
| 768–1199px | 220px rail, 28px content padding; PMRL details stay in one readable column |
| <768px | No fixed rail: replace with a labeled paper selector above the content; 16px outer padding; single-column PMRL details; structured comparison stacks by paper; tabs may scroll within their own row |

The narrowest supported Results viewport is **342px**. Verify that long findings and metric reasons wrap within their card or table container and that `document.documentElement.scrollWidth` does not exceed the viewport width. Horizontal scrolling, where necessary for a wide metric matrix, belongs to its bounded container only.

On narrow phones use a 40–48px Landing headline via a mobile override, a 120px minimum composer, and full-width Pipeline panel with 16px padding. No horizontal **page** scroll. Keep touch targets at least 44×44px, visible focus (2px `--accent` with offset), meaningful headings and labels, keyboard-operable tabs/disclosures/attachment removal, and focus management: composer → Pipeline heading after submit → Results heading after completion. Stage changes may be announced politely; don't make an incessant screen-reader ticker. Keep error messages associated with their triggering control or stage. Verify text contrast; never use muted metadata color for essential reading text or rely on status color alone.

## 9. Motion contract

| Event | Duration and property |
| --- | --- |
| Landing → Pipeline | 180ms ease-out; opacity plus at most 4–6px/scale .985 |
| New/updated stage | 140ms ease-out; opacity + translateY(3px) only on first appearance |
| Pipeline → Results | 150–180ms fade of panel; results enter without a large slide |
| Popover / details | 120ms ease-out; opacity |
| Source selection | 120ms background/text-color transition |

Running dot may pulse softly; no bounce, animated gradient, giant slides, full-screen skeleton, perpetual typing dots, or decorative AI orb. Respect `prefers-reduced-motion: reduce` by removing pulse/transforms and nonessential transitions.

## 10. Hard anti-patterns and acceptance checks

**Never:** build a home dashboard; keep a permanent Landing header/stepper; show all ten graph nodes as rows; turn Evaluate↔Refine into repeated task rows; claim a direct PDF was discovered through arXiv search; show benchmark merely because multiple papers exist; expose raw chain-of-thought; claim heuristic GitHub matches are official; show a confidence/relevance percentage without data; treat a memory checkpoint as durable; use glassmorphism, gradients, cold default gray, loud shadows, colorful cards, busy badges, 11px reading paragraphs, or library-default styling; wrap every paragraph in a card; introduce features not represented by the workflow.

Before declaring the UI ready, check these observable cases with real or fixture-backed event payloads:

1. Empty Landing, keyboard submission, valid attachment and validation errors.
2. Topic search with zero, one, and two refinements: still one Discover row; real counts only.
3. Direct arXiv/PDF: no search animation, no fake candidate count; explicit supplied-source reading.
4. Single paper: PMRL + report, no Comparison; multiple papers: Comparison only if artifact exists.
5. Missing repository/BibTeX or PDF evidence: no dead links, invented metadata, or fake citations.
6. Failure at PDF read, provider/`write_notes`, and report: specific error and truthful recovery; no stuck running state.
7. Out-of-order/duplicate events, reconnect/reload, and repeated submit: no duplicated stages or false resume.
8. Desktop/tablet/mobile, keyboard navigation, focus return, readable contrast, and reduced motion.

Success means a user can answer: **What is Research Scout doing now? What did it actually find? Which outputs are final, and where did they come from?** The interface should feel like a research notebook and editorial reading surface, not a chatbot, SaaS dashboard, or developer console.
