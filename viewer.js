(() => {
  const fileInput = document.getElementById("file-input");
  const fileNameEl = document.getElementById("file-name");
  const filterInput = document.getElementById("filter-input");
  const dropZone = document.getElementById("drop-zone");
  const statusEl = document.getElementById("status");
  const runMetaEl = document.getElementById("run-meta");
  const stepsEl = document.getElementById("steps");
  const errorsEl = document.getElementById("parse-errors");
  const expandAllBtn = document.getElementById("expand-all");
  const collapseAllBtn = document.getElementById("collapse-all");
  const clearBtn = document.getElementById("clear-view");

  const emptyState = "Load a log to see steps and tool activity.";

  const formatBytes = (bytes) => {
    if (!Number.isFinite(bytes)) return "";
    const units = ["B", "KB", "MB", "GB"];
    let value = bytes;
    let i = 0;
    while (value >= 1024 && i < units.length - 1) {
      value /= 1024;
      i += 1;
    }
    return `${value.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
  };

  const safeText = (value) => {
    if (value === null || value === undefined) return "";
    if (typeof value === "string") return value;
    try {
      return JSON.stringify(value, null, 2);
    } catch (err) {
      return String(value);
    }
  };

  const formatTimestamp = (seconds) => {
    if (!Number.isFinite(seconds)) return "";
    const date = new Date(seconds * 1000);
    return date.toLocaleString();
  };

  const formatTokens = (tokens) => {
    if (!tokens || typeof tokens !== "object") return "";
    const total = tokens.total_tokens ?? tokens.output_tokens ?? tokens.input_tokens;
    if (total === undefined) return "";
    const parts = [];
    if (tokens.prompt_tokens !== undefined) parts.push(`p${tokens.prompt_tokens}`);
    if (tokens.completion_tokens !== undefined) parts.push(`c${tokens.completion_tokens}`);
    if (tokens.total_tokens !== undefined && parts.length === 0) {
      return `tok ${tokens.total_tokens}`;
    }
    if (parts.length) {
      return `tok ${tokens.total_tokens ?? total} ${parts.join(" ")}`;
    }
    return `tok ${total}`;
  };

  const createEl = (tag, className, text) => {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text !== undefined) el.textContent = text;
    return el;
  };

  const truncateOneLine = (value, max = 140) => {
    if (!value) return "";
    const text = String(value).replace(/\s+/g, " ").trim();
    if (text.length <= max) return text;
    return `${text.slice(0, max - 1)}…`;
  };

  const applyFilter = () => {
    if (!filterInput) return;
    const term = filterInput.value.trim().toLowerCase();
    const stepNodes = stepsEl.querySelectorAll("details.step");
    stepNodes.forEach((node) => {
      if (!term) {
        node.classList.remove("hidden");
        return;
      }
      const haystack = (node.dataset.search || "").toLowerCase();
      node.classList.toggle("hidden", !haystack.includes(term));
    });
  };

  const createMetaPill = (label, value) => {
    const pill = createEl("div", "meta-pill");
    pill.title = `${label}: ${value}`;
    pill.append(createEl("span", "label", label));
    pill.append(createEl("span", "value", value));
    return pill;
  };

  const createMetaItem = (label, value, isMono = false) => {
    const item = createEl("div", "meta-item");
    item.append(createEl("div", "label", label));
    if (isMono) {
      const pre = createEl("div", "mono");
      pre.textContent = value;
      item.append(pre);
    } else {
      item.append(createEl("div", "value", value));
    }
    return item;
  };

  const clearView = () => {
    runMetaEl.innerHTML = "";
    stepsEl.innerHTML = "";
    errorsEl.innerHTML = "";
    errorsEl.classList.add("hidden");
    statusEl.textContent = emptyState;
    if (fileNameEl) fileNameEl.textContent = "No file loaded";
    if (filterInput) filterInput.value = "";
  };

  const parseJsonl = (text) => {
    const events = [];
    const errors = [];
    const lines = text.split(/\r?\n/);
    lines.forEach((line, index) => {
      const trimmed = line.trim();
      if (!trimmed) return;
      try {
        events.push(JSON.parse(trimmed));
      } catch (err) {
        errors.push({
          line: index + 1,
          message: err.message || String(err),
          value: trimmed.slice(0, 2000),
        });
      }
    });
    return { events, errors, lineCount: lines.length };
  };

  const groupByStep = (events) => {
    const steps = new Map();
    let runStart = null;
    let runEnd = null;

    events.forEach((event, index) => {
      if (event.type === "run_start") runStart = event;
      if (event.type === "run_end") runEnd = event;
      const step = event.step;
      if (!step) return;
      const agentName = event.agent || step.agent || "agent";
      const key = step.global !== undefined
        ? String(step.global)
        : `${agentName}:${step.local ?? index}`;
      if (!steps.has(key)) {
        steps.set(key, {
          key,
          local: step.local ?? null,
          global: step.global ?? null,
          agent: agentName,
          events: [],
          order: index,
        });
      }
      steps.get(key).events.push(event);
    });

    const orderedSteps = Array.from(steps.values()).sort((a, b) => {
      const aGlobal = a.global ?? Infinity;
      const bGlobal = b.global ?? Infinity;
      if (aGlobal !== bGlobal) return aGlobal - bGlobal;
      return a.order - b.order;
    });

    return { orderedSteps, runStart, runEnd };
  };

  const renderRunMeta = ({ runStart, runEnd, events, errors, fileInfo, lineCount, stepCount }) => {
    runMetaEl.innerHTML = "";

    const header = createEl("div", "section-title", "Run metadata");
    const row = createEl("div", "meta-row");

    const runId = runStart?.run_id || runEnd?.run_id || "";
    if (runId) row.append(createMetaPill("run_id", runId));
    if (runStart?.agent) row.append(createMetaPill("agent", runStart.agent));
    if (runStart?.model) row.append(createMetaPill("model", runStart.model));
    if (runStart?.temperature !== undefined) row.append(createMetaPill("temp", String(runStart.temperature)));
    if (runStart?.max_steps !== undefined) row.append(createMetaPill("max_steps", String(runStart.max_steps)));
    if (runStart?.reset !== undefined) row.append(createMetaPill("reset", String(runStart.reset)));
    if (runStart?.resume_from) row.append(createMetaPill("resume_from", runStart.resume_from));
    if (runStart && runStart.resume_step !== undefined) {
      row.append(createMetaPill("resume_step", String(runStart.resume_step)));
    }
    if (runStart?.ts) row.append(createMetaPill("started", formatTimestamp(runStart.ts)));
    if (runEnd?.ts) row.append(createMetaPill("ended", formatTimestamp(runEnd.ts)));
    if (runEnd?.status) row.append(createMetaPill("status", runEnd.status));
    if (fileInfo?.name) row.append(createMetaPill("file", fileInfo.name));
    if (fileInfo?.size !== undefined) row.append(createMetaPill("size", formatBytes(fileInfo.size)));
    row.append(createMetaPill("events", String(events.length)));
    row.append(createMetaPill("steps", String(stepCount)));
    row.append(createMetaPill("lines", String(lineCount)));
    if (errors.length) row.append(createMetaPill("parse_errors", String(errors.length)));

    runMetaEl.append(header, row);

    if (runStart?.task) {
      const taskWrap = createEl("div", "task-line");
      taskWrap.append(createEl("div", "label", "Task"));
      const taskContent = createEl("div", "task-content");
      taskContent.textContent = safeText(runStart.task);
      taskWrap.append(taskContent);
      runMetaEl.append(taskWrap);
    }
  };

  const renderErrors = (errors) => {
    if (!errors.length) {
      errorsEl.classList.add("hidden");
      errorsEl.innerHTML = "";
      return;
    }

    errorsEl.classList.remove("hidden");
    errorsEl.innerHTML = "";
    errorsEl.append(createEl("div", "section-title", "Parse errors"));

    const list = createEl("div");
    errors.slice(0, 20).forEach((error) => {
      const block = createEl("div", "tool-block");
      block.append(createEl("div", "tool-head", `Line ${error.line}`));
      const pre = createEl("pre", "mono");
      pre.textContent = `${error.message}\n${error.value}`;
      block.append(pre);
      list.append(block);
    });
    if (errors.length > 20) {
      list.append(createEl("div", "value", `Showing first 20 of ${errors.length} errors.`));
    }
    errorsEl.append(list);
  };

  const renderSteps = (orderedSteps) => {
    stepsEl.innerHTML = "";
    if (!orderedSteps.length) {
      stepsEl.append(createEl("div", "panel", "No step events found."));
      return;
    }

    orderedSteps.forEach((stepData) => {
      const detail = createEl("details", "step fade-in");
      const summary = createEl("summary");

      const stepLabel = stepData.global !== null
        ? `Step ${stepData.global}`
        : `Step ${stepData.local ?? "?"}`;
      summary.append(createEl("span", "step-id", stepLabel));

      if (stepData.agent) {
        summary.append(createEl("span", "chip", stepData.agent));
      }
      if (stepData.local !== null && stepData.global !== null && stepData.local !== stepData.global) {
        summary.append(createEl("span", "chip", `local ${stepData.local}`));
      }

      const events = stepData.events;
      const request = events.find((e) => e.type === "request");
      const assistant = events.find((e) => e.type === "assistant_message");
      const stepEnd = events.find((e) => e.type === "step_end");
      const toolCalls = events.filter((e) => e.type === "tool_call");
      const toolResults = events.filter((e) => e.type === "tool_result");
      const userMessages = events.filter((e) => e.type === "user_message");

      const previewText = truncateOneLine(assistant?.content);
      if (previewText) summary.append(createEl("span", "step-preview", previewText));

      const toolNameSummary = (() => {
        const names = [];
        const seen = new Set();
        toolCalls.forEach((call) => {
          const name = call?.name || call?.function?.name;
          if (name && !seen.has(name)) {
            names.push(name);
            seen.add(name);
          }
        });
        if (!names.length) return "";
        const maxShown = 2;
        let label = names.slice(0, maxShown).join(", ");
        const remaining = names.length - maxShown;
        if (remaining > 0) label += ` +${remaining}`;
        return names.length === 1 ? `tool ${label}` : `tools ${label}`;
      })();

      const status = stepEnd?.status || "unknown";
      const statusChip = createEl("span", "chip");
      statusChip.textContent = status;
      if (status === "completed") statusChip.classList.add("completed");
      if (status === "no_tool_call") statusChip.classList.add("no-tool");
      if (status === "exhausted") statusChip.classList.add("exhausted");
      summary.append(statusChip);

      if (toolNameSummary) summary.append(createEl("span", "chip", toolNameSummary));
      const hasError = toolResults.some((result) => result?.error);
      if (hasError) summary.append(createEl("span", "chip alert", "error"));
      if (stepEnd?.wall_time_sec !== undefined) {
        summary.append(createEl("span", "chip", `${stepEnd.wall_time_sec.toFixed(3)}s`));
      }
      const tokenInfo = formatTokens(stepEnd?.tokens);
      if (tokenInfo) summary.append(createEl("span", "chip", tokenInfo));

      detail.append(summary);

      const searchParts = [
        stepLabel,
        String(stepData.global ?? ""),
        stepData.agent ?? "",
        status,
        toolNameSummary,
        previewText,
        truncateOneLine(assistant?.content, 400),
        safeText(toolCalls.map((call) => call.name || call?.function?.name || "").join(" ")),
        safeText(toolResults.map((result) => truncateOneLine(result?.result || "", 80)).join(" ")),
      ];
      detail.dataset.search = searchParts.filter(Boolean).join(" ");

      const body = createEl("div");

      if (request) {
        body.append(createEl("div", "section-title", "Request"));
        const requestRow = createEl("div", "meta-row");
        if (request.model) requestRow.append(createMetaPill("model", request.model));
        if (request.temperature !== undefined) {
          requestRow.append(createMetaPill("temp", String(request.temperature)));
        }
        if (request.messages !== undefined) requestRow.append(createMetaPill("messages", String(request.messages)));
        body.append(requestRow);

        if (request.tool_schema_hash) {
          const details = createEl("details", "tool-block");
          const summaryLine = createEl("summary", "tool-head", "tool_schema_hash");
          details.append(summaryLine);
          const pre = createEl("pre", "mono");
          pre.textContent = safeText(request.tool_schema_hash);
          details.append(pre);
          body.append(details);
        }

        if (request.tool_names) {
          const details = createEl("details", "tool-block");
          const summaryLine = createEl("summary", "tool-head", "tool_names");
          details.append(summaryLine);
          const pre = createEl("pre", "mono");
          pre.textContent = safeText(request.tool_names);
          details.append(pre);
          body.append(details);
        }
      }

      body.append(createEl("div", "section-title", "Assistant"));
      const assistantBlock = createEl("pre", "mono");
      assistantBlock.textContent = safeText(assistant?.content || "<no content>");
      body.append(assistantBlock);

      if (toolCalls.length || toolResults.length) {
        body.append(createEl("div", "section-title", "Tools"));
        const resultsById = new Map();
        toolResults.forEach((result) => {
          if (result.tool_call_id) resultsById.set(result.tool_call_id, result);
        });

        toolCalls.forEach((call, index) => {
          const toolBlock = createEl("details", "tool-block");
          const toolName = call.name || call?.function?.name || "tool";
          const result = call.tool_call_id ? resultsById.get(call.tool_call_id) : null;
          const resultPreview = truncateOneLine(result?.result || result?.content || "");
          const summaryLine = createEl(
            "summary",
            "tool-head",
            `${index + 1}. ${toolName}${resultPreview ? ` — ${resultPreview}` : ""}`
          );
          toolBlock.append(summaryLine);

          const argLabel = createEl("div", "tool-caption", "Arguments");
          const argPre = createEl("pre", "mono");
          argPre.textContent = safeText(call.arguments ?? call?.function?.arguments ?? {});
          toolBlock.append(argLabel, argPre);

          if (result) {
            const resLabel = createEl("div", "tool-caption", "Result");
            const resPre = createEl("pre", "mono");
            resPre.textContent = safeText(result.result ?? result.content ?? "");
            toolBlock.append(resLabel, resPre);
          }
          body.append(toolBlock);
        });
      }

      if (userMessages.length) {
        body.append(createEl("div", "section-title", "User parts"));
        userMessages.forEach((msg) => {
          const pre = createEl("pre", "mono");
          pre.textContent = safeText(msg.content);
          body.append(pre);
        });
      }

      detail.append(body);
      stepsEl.append(detail);
    });
  };

  const render = (events, errors, fileInfo, lineCount) => {
    const { orderedSteps, runStart, runEnd } = groupByStep(events);
    renderRunMeta({
      runStart,
      runEnd,
      events,
      errors,
      fileInfo,
      lineCount,
      stepCount: orderedSteps.length,
    });
    renderErrors(errors);
    renderSteps(orderedSteps);
    applyFilter();
  };

  const loadFile = async (file) => {
    if (!file) return;
    statusEl.textContent = `Loading ${file.name}...`;
    const text = await file.text();
    const { events, errors, lineCount } = parseJsonl(text);
    render(events, errors, { name: file.name, size: file.size }, lineCount);
    statusEl.textContent = `Loaded ${events.length} events from ${file.name}.`;
    if (fileNameEl) fileNameEl.textContent = file.name;
  };

  fileInput.addEventListener("change", (event) => {
    const file = event.target.files[0];
    loadFile(file);
  });


  dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropZone.classList.add("is-over");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("is-over");
  });

  dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-over");
    const file = event.dataTransfer.files[0];
    loadFile(file);
  });

  dropZone.addEventListener("click", () => {
    fileInput.click();
  });

  expandAllBtn.addEventListener("click", () => {
    stepsEl.querySelectorAll("details").forEach((detail) => {
      detail.open = true;
    });
  });

  collapseAllBtn.addEventListener("click", () => {
    stepsEl.querySelectorAll("details").forEach((detail) => {
      detail.open = false;
    });
  });

  clearBtn.addEventListener("click", () => {
    fileInput.value = "";
    clearView();
  });

  if (filterInput) {
    filterInput.addEventListener("input", applyFilter);
  }

  clearView();
})();
