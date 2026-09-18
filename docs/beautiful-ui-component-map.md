# Beautiful UI adoption map

Research Scout borrows small interaction patterns from
[Beautiful UI](https://www.beautifului.dev/); it does not ship the gallery's
chat shell, dashboard, or demo data. The application has three successive
states: a minimal Landing composer, a semantic LangGraph Pipeline, and the
PMRL Results workspace. Every visible value comes from the run snapshot,
trace, or finalized artifact. See [`frontend/NOTICE`](../frontend/NOTICE) for
the MIT copyright and permission notice.

| Primitive | Decision | Shipped location | Required backend truth |
| --- | --- | --- | --- |
| [Prompt Bar](https://www.beautifului.dev/r/prompt-bar.json) | USE | `frontend/src/pages/LandingPage.tsx` | Topic text, arXiv references, validated PDF uploads, and the `/runs` response |
| [Loading State](https://www.beautifului.dev/r/loading-state.json) | USE, restrained | `frontend/src/pages/RunPage.tsx` | A real running snapshot; measured elapsed time is allowed, but no time estimate, percentage, shimmer, or fake progress |
| [Task Rows](https://www.beautifului.dev/r/task-rows.json) | USE | `frontend/src/pages/RunPage.tsx` and `lib/runAdapter.ts` | Real node lifecycle/state events mapped to semantic stages |
| [Tool Chips](https://www.beautifului.dev/r/tool-chips.json) | USE, factual only | Pipeline activity chips | Counts from allowlisted `TraceFacts`; a missing fact stays absent |
| [Thinking](https://www.beautifului.dev/r/thinking-state.json) | ADAPT as Activity | Pipeline and Results Activity disclosures | Operational lifecycle events only; never model reasoning or prompt text |
| [Context Cards](https://www.beautifului.dev/r/context-cards.json) | CONDITIONAL | `frontend/src/components/results/ResultsWorkspace.tsx` | Finalized PMRL fields or genuine source excerpts; no invented evidence |
| [Sidebar Nav](https://www.beautifului.dev/r/sidebar-nav.json) | DO NOT USE | — | The minimal Landing has no navbar, history, or dashboard shell |
| [Chat](https://www.beautifului.dev/r/chat-composer.json) | DO NOT USE | — | Conversation UI is outside the shipped three-state contract |
| [Agent Screen](https://github.com/slev12397/beautiful-ui/blob/main/components/primitives/AgentScreen.tsx) | NOT RELEVANT | — | The workflow has no browser/computer-use agent |
| [Approval Card](https://www.beautifului.dev/r/approval-card.json) | DISABLED | — | No LangGraph interrupt/resume contract exists |

## Adaptation rules

- Remove gallery demo timers, scripted typing, sample content, fake tool calls,
  and progress percentages. The adapter in `frontend/src/lib/runAdapter.ts`
  orders and de-duplicates trace events by sequence before deriving UI state.
- Keep conditional stages truthful. Direct arXiv/PDF inputs skip discovery;
  comparison appears only for an explicit direct comparison or a real compare
  trace and finalized benchmark. The Results workspace hides tabs for absent
  artifacts.
- Keep public activity concise and safe. Strip private paths, credentials,
  prompt internals, and raw model output; expose only semantic lifecycle
  summaries and allowlisted counts.
- Use the existing Lucide icon set. The gallery's commercial
  `@central-icons-react` dependency and optional gallery dependencies are not
  adopted.
- Markdown is rendered through the existing safe renderer without raw HTML.
  Missing PMRL fields, repositories, BibTeX, and evidence remain explicitly
  unavailable rather than being filled with guesses.

Sources: [catalog](https://www.beautifului.dev/),
[registry](https://www.beautifului.dev/r/registry.json),
[source tree](https://github.com/slev12397/beautiful-ui/tree/main/components/primitives),
[license](https://www.beautifului.dev/license).
