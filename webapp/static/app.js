const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const state = {
  scenePayload: null,
  health: null,
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.hidden = false;
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => {
    toast.hidden = true;
  }, 5200);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(payload.error || `요청 실패 (${response.status})`);
    error.code = payload.code;
    throw error;
  }
  return payload;
}

function formPayload(form) {
  const data = new FormData(form);
  return Object.fromEntries(data.entries());
}

function thumbnailFor(url) {
  try {
    const parsed = new URL(url);
    let id = "";
    if (parsed.hostname.endsWith("youtu.be")) id = parsed.pathname.slice(1);
    if (parsed.pathname.includes("/shorts/")) id = parsed.pathname.split("/shorts/")[1].split("/")[0];
    if (!id) id = parsed.searchParams.get("v") || "";
    return id ? `https://i.ytimg.com/vi/${id}/hqdefault.jpg` : "";
  } catch {
    return "";
  }
}

function switchTab(name) {
  $$(".tab-button").forEach((button) => {
    const active = button.dataset.tab === name;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", String(active));
  });
  $$(".tab-panel").forEach((panel) => {
    const active = panel.id === `${name}-panel`;
    panel.classList.toggle("is-active", active);
    panel.hidden = !active;
  });
}

function updateJudgeThumbnail() {
  const direct = $("#judge-thumbnail-url").value.trim();
  const derived = thumbnailFor($("#judge-video-url").value.trim());
  const url = direct || derived;
  const shell = $(".thumbnail-shell");
  const image = $("#judge-thumbnail-preview");
  if (!url) {
    shell.classList.remove("has-image");
    image.removeAttribute("src");
    return;
  }
  image.src = url;
  image.alt = "평가 대상 썸네일";
  image.onload = () => shell.classList.add("has-image");
  image.onerror = () => shell.classList.remove("has-image");
}

function axisRow(label, value, weight) {
  const percent = Math.max(0, Math.min(100, Number(value) * 25));
  return `
    <div class="axis-row">
      <label>${escapeHtml(label)} <small>${escapeHtml(weight)}</small></label>
      <div class="axis-track"><span style="width:${percent}%"></span></div>
      <strong>${escapeHtml(value)}/4</strong>
    </div>
  `;
}

function renderJudge(result) {
  const live = result.mode === "live_llm";
  const strengths = result.strengths?.length ? result.strengths : ["뚜렷한 강점 신호 없음"];
  const risks = result.risks?.length ? result.risks : ["뚜렷한 감점 신호 없음"];
  $("#judge-result").innerHTML = `
    <div class="judge-result-shell">
      <div class="result-topline">
        <div class="score-block">
          <div class="score-value">${escapeHtml(result.editorial_success_score)}<small>/100</small></div>
          <div class="score-label">EDITORIAL SUCCESS</div>
        </div>
        <div class="result-thumbnail">
          <img src="${escapeHtml(result.thumbnail_url)}" alt="평가 대상 썸네일" />
          <div class="result-tags">
            <span class="tag ${live ? "live" : "preview"}">${live ? "LIVE LLM" : "OFFLINE PREVIEW"}</span>
            <span class="tag">${escapeHtml(result.model)}</span>
            <span class="tag">신뢰도 ${escapeHtml(result.confidence_1_5)}/5</span>
          </div>
        </div>
      </div>
      <div class="axis-list">
        ${axisRow("변화·반전", result.change_or_surprise_0_4, "40%")}
        ${axisRow("제목 패키징", result.title_packaging_0_4, "15%")}
        ${axisRow("썸네일 패키징", result.thumbnail_packaging_0_4, "45%")}
      </div>
      <div class="evidence-block">
        <h2>판단 근거</h2>
        <p>${escapeHtml(result.evidence_first)}</p>
        <div class="signal-columns">
          <div class="signal-column">
            <h3>강점</h3>
            <ul>${strengths.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
          </div>
          <div class="signal-column risk">
            <h3>주의 신호</h3>
            <ul>${risks.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
          </div>
        </div>
      </div>
      ${result.warning ? `<div class="warning-banner">${escapeHtml(result.warning)}</div>` : ""}
    </div>
  `;
}

function renderCandidate(candidate, thumbnailUrl) {
  const supplement = candidate.selection_role === "judge_supplement";
  return `
    <article class="candidate-card ${supplement ? "is-supplement" : ""}">
      <a class="candidate-image" href="${escapeHtml(candidate.watch_url)}" target="_blank" rel="noreferrer">
        <img src="${escapeHtml(thumbnailUrl)}" alt="" />
        <span class="rank-badge">${escapeHtml(candidate.rank)}</span>
      </a>
      <div class="candidate-content">
        <h2>${escapeHtml(candidate.generated_title)}</h2>
        <p class="candidate-time">${escapeHtml(candidate.start_time)} – ${escapeHtml(candidate.end_time)} · ${escapeHtml(candidate.duration_sec)}초</p>
        <div class="score-pair">
          <div><span>구조 점수</span><strong>${escapeHtml(candidate.structural_score)}</strong></div>
          <div><span>Judge 점수</span><strong>${escapeHtml(candidate.judge_score)}</strong></div>
        </div>
        <span class="candidate-role">${supplement ? "JUDGE 보강" : "AC ANCHOR"}</span>
      </div>
    </article>
  `;
}

function renderGenerator(result) {
  const thumbnailUrl = thumbnailFor(result.video_url);
  const mode = result.mode === "live_llm" ? "LIVE LLM" : "OFFLINE PREVIEW";
  $("#generator-result").innerHTML = `
    <div class="pipeline-summary">
      <div>
        <p class="eyebrow">FINAL TOP5 · ${escapeHtml(mode)}</p>
        <h1>서로 다른 사건을 담은 후보</h1>
        <p>${escapeHtml(result.data_source)} · ${escapeHtml(result.scene_count)} scenes · ${escapeHtml(result.model)}</p>
      </div>
      <div class="summary-metrics">
        <div class="summary-metric"><strong>${escapeHtml(result.compressed_candidate_count)}</strong><span>압축 후보</span></div>
        <div class="summary-metric"><strong>4+1</strong><span>AC + Judge</span></div>
        <div class="summary-metric"><strong>58%</strong><span>중복 한계</span></div>
      </div>
    </div>
    ${result.warning ? `<div class="warning-banner">${escapeHtml(result.warning)}</div>` : ""}
    <div class="candidate-list">
      ${result.final_candidates.map((candidate) => renderCandidate(candidate, thumbnailUrl)).join("")}
    </div>
  `;
}

async function loadHealth() {
  const status = $("#system-status");
  try {
    state.health = await api("/api/health");
    status.classList.add("is-ready");
    status.innerHTML = `<span class="status-dot"></span>${state.health.openai_ready ? `LLM · ${escapeHtml(state.health.model)}` : "프리뷰 모드"} · ${state.health.library_count} videos`;
  } catch {
    status.innerHTML = `<span class="status-dot"></span>서버 연결 실패`;
  }
}

$$(".tab-button").forEach((button) => {
  button.addEventListener("click", () => switchTab(button.dataset.tab));
});

$("#judge-video-url").addEventListener("input", updateJudgeThumbnail);
$("#judge-thumbnail-url").addEventListener("input", updateJudgeThumbnail);

$("#judge-sample").addEventListener("click", () => {
  const form = $("#judge-form");
  form.elements.video_url.value = "https://www.youtube.com/shorts/BETA_DEMO01";
  form.elements.start_time.value = "8:41";
  form.elements.end_time.value = "9:30";
  form.elements.title.value = "마감 20초 전 들어온 마지막 주문";
  form.elements.description.value = "마감 직전 주문이 99개에 머물지만 마지막 손님이 등장해 목표 100개를 달성하고 모두 환호한다.";
  form.elements.transcript.value = "[8:41-8:55] 첫 손님이 한입 먹은 뒤 아무 말도 하지 않습니다.\n[8:55-9:05] 왜 아무 말씀도 없으세요? 혹시 너무 매워요?\n[9:05-9:20] 생각보다 너무 맛있어서 놀랐어요. 하나 더 주세요.\n[9:20-9:30] 종료 20초를 남기고 목표를 달성했습니다.";
  form.elements.thumbnail_url.value = "";
  updateJudgeThumbnail();
});

$("#judge-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const result = $("#judge-result");
  $("#judge-empty").hidden = true;
  result.hidden = true;
  $("#judge-loading").hidden = false;
  try {
    const payload = formPayload(event.currentTarget);
    const response = await api("/api/judge", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    renderJudge(response);
    result.hidden = false;
  } catch (error) {
    $("#judge-empty").hidden = false;
    showToast(error.message);
  } finally {
    $("#judge-loading").hidden = true;
  }
});

$("#scene-file").addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    state.scenePayload = JSON.parse(await file.text());
    $("#scene-file-name").textContent = `${file.name} · 업로드 준비됨`;
  } catch {
    state.scenePayload = null;
    $("#scene-file-name").textContent = "JSON을 읽지 못했습니다";
    showToast("유효한 JSON 파일을 선택해 주세요.");
  }
});

$("#generator-sample").addEventListener("click", async () => {
  const form = $("#generator-form");
  form.elements.video_url.value = "https://www.youtube.com/watch?v=BETA_DEMO01";
  form.elements.vpick_asset_url.value = "";
  try {
    state.scenePayload = await api("/api/sample-scenes");
    $("#scene-file-name").textContent = "BETA_DEMO01_scenes.json · 샘플";
    showToast("샘플 Vpick 장면 12개를 불러왔습니다.");
  } catch (error) {
    showToast(error.message);
  }
});

$("#generator-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const result = $("#generator-result");
  $("#generator-empty").hidden = true;
  result.hidden = true;
  $("#generator-loading").hidden = false;
  try {
    const payload = { ...formPayload(event.currentTarget), scene_payload: state.scenePayload };
    const response = await api("/api/highlights", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    renderGenerator(response);
    result.hidden = false;
  } catch (error) {
    $("#generator-empty").hidden = false;
    showToast(error.message);
  } finally {
    $("#generator-loading").hidden = true;
  }
});

loadHealth();
