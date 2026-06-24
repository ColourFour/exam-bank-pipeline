const STORAGE_KEYS = {
  selectedClass: "examBankClassroom.selectedClassId",
  activeTab: "examBankClassroom.activeTab"
};

let state = {
  selectedClassId: localStorage.getItem(STORAGE_KEYS.selectedClass) || "",
  selectedAssignmentId: "",
  activeTab: localStorage.getItem(STORAGE_KEYS.activeTab) || "overview",
  classes: [],
  classDetail: null,
  rosterRows: [],
  assignmentDetail: null,
  modalAction: null
};

const $ = (id) => document.getElementById(id);

function setActiveTab(tab) {
  state.activeTab = tab;
  localStorage.setItem(STORAGE_KEYS.activeTab, tab);
  renderTabs();
}

function showToast(value) {
  const toast = $("toast");
  toast.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  toast.hidden = false;
  window.clearTimeout(showToast.timeout);
  showToast.timeout = window.setTimeout(() => {
    toast.hidden = true;
  }, 4500);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const text = await response.text();
  let payload;
  try {
    payload = JSON.parse(text);
  } catch {
    payload = { raw: text };
  }
  if (!response.ok) {
    showToast(payload);
    throw new Error(payload.message || payload.error || response.statusText);
  }
  return payload;
}

function hide(id) {
  $(id).hidden = true;
}

function show(id) {
  $(id).hidden = false;
}

function openAddClassModal() {
  show("addClassPanel");
  window.setTimeout(() => {
    const firstInput = $("addClassPanel").querySelector("input");
    if (firstInput) firstInput.focus();
  }, 0);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function assignmentLabel(assignment, fallback) {
  const title = String((assignment && assignment.title) || fallback || "").trim();
  if (!title) return "assignment";
  return title.toLowerCase().startsWith("assignment:") ? title : `assignment: ${title}`;
}

function isValidEmail(value) {
  return !value || /^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(value);
}

function badgeClass(status) {
  const normalized = String(status || "").toLowerCase();
  if (["active", "sent", "submitted", "scheduled"].includes(normalized)) return "success";
  if (["archived", "failed", "invalid", "missing"].includes(normalized)) return "danger";
  if (["draft", "not_sent", "not_scheduled"].includes(normalized)) return "warning";
  return "";
}

async function loadClasses() {
  const payload = await api("/api/classes");
  state.classes = payload.classes || [];
  $("classCount").textContent = `${state.classes.length} class${state.classes.length === 1 ? "" : "es"}`;
  renderClassList();
  if (state.selectedClassId && state.classes.some((item) => item.class_id === state.selectedClassId)) {
    await openClass(state.selectedClassId, { preserveTab: true });
  } else {
    renderEmptyState();
  }
}

function renderEmptyState() {
  $("emptyState").hidden = false;
  $("classWorkspace").hidden = true;
}

function renderClassList() {
  const root = $("classList");
  root.innerHTML = "";
  if (!state.classes.length) {
    root.innerHTML = `<div class="empty-state compact"><p>No classes yet.</p></div>`;
    return;
  }
  for (const item of state.classes) {
    const selected = state.selectedClassId === item.class_id;
    const card = document.createElement("button");
    card.type = "button";
    card.className = `class-nav-item${selected ? " selected" : ""}`;
    card.dataset.openClass = item.class_id;
    card.innerHTML = `
      <span class="class-nav-top">
        <span>
          <span class="class-title">${escapeHtml(item.display_name || item.class_id)}</span>
          <span class="class-meta">${escapeHtml(item.course_id || "course")}</span>
        </span>
        <span class="student-count"><strong>${item.student_count || 0}</strong> students</span>
      </span>
      <span class="class-count-grid">
        <span><strong>${item.active_assignment_count || 0}</strong><small>active</small></span>
        <span><strong>${item.draft_assignment_count || 0}</strong><small>draft</small></span>
        <span><strong>${item.archived_assignment_count || 0}</strong><small>archived</small></span>
      </span>
    `;
    root.appendChild(card);
  }
}

async function openClass(classId, options = {}) {
  state.selectedClassId = classId;
  localStorage.setItem(STORAGE_KEYS.selectedClass, classId);
  const payload = await api(`/api/classes/${encodeURIComponent(classId)}`);
  state.classDetail = payload;
  state.rosterRows = [];
  $("emptyState").hidden = true;
  $("classWorkspace").hidden = false;
  $("classTitle").textContent = payload.class.display_name || classId;
  $("classCourseMeta").textContent = `${payload.class.course_id || "course"} - ${payload.class.teacher_email || ""}`;
  renderClassSummary(payload);
  renderAssignments(payload.assignments || []);
  renderReports([]);
  renderClassList();
  if (!options.preserveTab) setActiveTab("overview");
  renderTabs();
}

function renderClassSummary(payload) {
  const counts = payload.assignment_counts || {};
  $("classSummary").innerHTML = `
    <div class="metric"><strong>${payload.roster_summary.active_students}</strong><span>active students</span></div>
    <div class="metric"><strong>${counts.active || 0}</strong><span>active assignments</span></div>
    <div class="metric"><strong>${counts.draft || 0}</strong><span>draft assignments</span></div>
    <div class="metric"><strong>${counts.archived || 0}</strong><span>archived assignments</span></div>
  `;
  $("rosterSummary").textContent = `${payload.roster_summary.active_students} active students, ${payload.roster_summary.with_email} with email.`;
}

function renderTabs() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === state.activeTab);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.hidden = !panel.id.startsWith(state.activeTab);
  });
}

function renderAssignments(assignments) {
  renderAssignmentCards($("assignmentList"), assignments, { includeActions: true });
  renderAssignmentCards($("overviewAssignments"), assignments.slice(0, 4), { includeActions: true });
}

function renderAssignmentCards(root, assignments, { includeActions }) {
  root.innerHTML = "";
  if (!assignments.length) {
    root.innerHTML = `<div class="empty-state compact"><p>No assignments yet.</p></div>`;
    return;
  }
  for (const item of assignments) {
    const selected = state.selectedAssignmentId === item.assignment_id;
    const card = document.createElement("article");
    card.className = `assignment-card${selected ? " selected" : ""}`;
    card.innerHTML = `
      <div class="assignment-card-head">
        <div>
          <div class="assignment-title">${escapeHtml(assignmentLabel(item, item.assignment_id))}</div>
          <div class="class-meta">${escapeHtml(item.assignment_id)}</div>
        </div>
        <span class="badge ${badgeClass(item.assignment_status || item.email_status)}">${escapeHtml(item.assignment_status || item.email_status || "")}</span>
      </div>
      <div class="metric-row">
        <div class="metric"><strong>${escapeHtml(item.send_at || "")}</strong><span>send</span></div>
        <div class="metric"><strong>${escapeHtml(item.due_at || "")}</strong><span>due</span></div>
        <div class="metric"><strong>${item.submitted_count}</strong><span>submitted</span></div>
        <div class="metric"><strong>${item.missing_count}</strong><span>missing</span></div>
      </div>
      ${
        includeActions
          ? `<div class="button-row">
              <button class="button secondary" type="button" data-open-assignment="${escapeHtml(item.assignment_id)}">Open</button>
              <button class="button secondary" type="button" data-assignment-tab="${escapeHtml(item.assignment_id)}" data-tab-target="submissions">Upload submissions</button>
              <button class="button primary" type="button" data-assignment-tab="${escapeHtml(item.assignment_id)}" data-tab-target="email">Email</button>
            </div>`
          : ""
      }
    `;
    root.appendChild(card);
  }
}

async function editRoster(classId = state.selectedClassId) {
  state.selectedClassId = classId;
  const payload = await api(`/api/classes/${encodeURIComponent(classId)}/roster`);
  state.rosterRows = payload.rows || [];
  renderRoster();
  $("rosterEditor").hidden = false;
  $("toggleRosterBtn").textContent = "Hide roster";
}

function renderRoster() {
  const tbody = $("rosterTable").querySelector("tbody");
  tbody.innerHTML = "";
  for (const row of state.rosterRows) addRosterRow(row);
}

function addRosterRow(row = {}) {
  const tbody = $("rosterTable").querySelector("tbody");
  const tr = document.createElement("tr");
  if (!isValidEmail(row.email || "")) tr.classList.add("invalid-email");
  tr.innerHTML = `
    <td><input name="student_id" value="${escapeHtml(row.student_id || "")}"></td>
    <td><input name="display_name" value="${escapeHtml(row.display_name || "")}"></td>
    <td><input name="email" value="${escapeHtml(row.email || "")}"></td>
    <td><input name="active" value="${escapeHtml(row.active || "true")}"></td>
    <td><button class="button secondary" type="button" data-remove-row>Deactivate</button></td>
  `;
  tbody.appendChild(tr);
}

function rosterRows() {
  return Array.from($("rosterTable").querySelectorAll("tbody tr")).map((tr) => {
    const row = {};
    tr.querySelectorAll("input").forEach((input) => {
      row[input.name] = input.value;
    });
    row.class_id = state.selectedClassId;
    return row;
  });
}

async function saveRoster() {
  const payload = await api(`/api/classes/${encodeURIComponent(state.selectedClassId)}/roster`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rows: rosterRows() })
  });
  showToast("Roster saved.");
  state.rosterRows = payload.rows || [];
  await openClass(state.selectedClassId, { preserveTab: true });
  renderRoster();
}

async function createClass(event) {
  event.preventDefault();
  const data = Object.fromEntries(new FormData(event.target).entries());
  const payload = await api("/api/classes", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data)
  });
  hide("addClassPanel");
  showToast("Class created.");
  await loadClasses();
  await openClass(payload.class.class_id);
}

async function createAssignment(event) {
  event.preventDefault();
  const form = new FormData(event.target);
  form.append("course_id", "p3");
  const payload = await api(`/api/classes/${encodeURIComponent(state.selectedClassId)}/assignments`, {
    method: "POST",
    body: form
  });
  hide("assignmentFormPanel");
  showToast("Assignment saved.");
  await openClass(state.selectedClassId, { preserveTab: true });
  await openAssignment(payload.assignment.assignment.assignment_id, "email");
}

async function openAssignment(assignmentId, tab = state.activeTab) {
  state.selectedAssignmentId = assignmentId;
  const payload = await api(`/api/classes/${encodeURIComponent(state.selectedClassId)}/assignments/${encodeURIComponent(assignmentId)}`);
  state.assignmentDetail = payload;
  $("assignmentDetailPanel").hidden = false;
  $("noAssignmentEmailState").hidden = true;
  $("assignmentTitle").textContent = assignmentLabel(payload.assignment, assignmentId);
  $("assignmentMeta").textContent = assignmentId;
  $("assignmentPdfLink").href = payload.assignment_pdf || "#";
  $("assignmentSummary").innerHTML = `
    <div class="metric"><strong>${escapeHtml(payload.assignment.send_at || "")}</strong><span>send</span></div>
    <div class="metric"><strong>${escapeHtml(payload.assignment.due_at || "")}</strong><span>due</span></div>
    <div class="metric"><strong>${payload.roster_count}</strong><span>roster count</span></div>
    <div class="metric"><strong>${payload.submission_status.submitted}</strong><span>submitted</span></div>
    <div class="metric"><strong>${payload.submission_status.missing}</strong><span>missing</span></div>
  `;
  $("submissionAssignmentSummary").textContent = `${assignmentLabel(payload.assignment, assignmentId)} selected.`;
  $("emailStatusBadge").textContent = payload.dispatch_status || "not sent";
  $("emailStatusBadge").className = `badge ${badgeClass(payload.dispatch_status)}`;
  renderAssignmentEmailPreview(payload.email_preview || {});
  await previewAck({ silent: true });
  renderReports(payload.reports || []);
  renderAssignments((state.classDetail && state.classDetail.assignments) || []);
  if (tab) setActiveTab(tab);
}

function renderAssignmentEmailPreview(preview) {
  const sentRecipients = Array.isArray(preview.sent_recipients) ? preview.sent_recipients : [];
  const sentList = sentRecipients
    .slice(0, 12)
    .map((row) => `${escapeHtml(row.student_id || "")} ${escapeHtml(row.email || "")} ${escapeHtml(row.sent_at || "")}`.trim())
    .join("<br>");
  $("assignmentEmailPreview").innerHTML = `
    ${
      preview.already_sent
        ? `<div class="preview-item warning full-width"><strong>Already sent</strong>This assignment email has already been sent to ${preview.sent_recipient_count || sentRecipients.length} student${(preview.sent_recipient_count || sentRecipients.length) === 1 ? "" : "s"}. Sending now will resend it.</div>`
        : ""
    }
    ${
      sentRecipients.length
        ? `<div class="preview-item full-width"><strong>Sent emails</strong>${sentList}${sentRecipients.length > 12 ? "<br>..." : ""}</div>`
        : ""
    }
    <div class="preview-item"><strong>Provider</strong>${escapeHtml(preview.provider || "Mail.app")}</div>
    <div class="preview-item"><strong>Recipients ready</strong>${preview.recipient_count || 0} / ${preview.roster_total || 0}</div>
    <div class="preview-item"><strong>Invalid emails</strong>${preview.invalid_email_count || 0}</div>
    <div class="preview-item"><strong>Missing emails</strong>${preview.missing_email_count || 0}</div>
    <div class="preview-item"><strong>Subject</strong>${escapeHtml(preview.subject || "")}</div>
    <div class="preview-item"><strong>PDF</strong>${preview.pdf_attached ? "attached" : "missing"}</div>
  `;
}

function renderReceiptPreview(payload) {
  $("receiptPreview").innerHTML = `
    <div class="preview-item"><strong>Eligible submitted students</strong>${payload.count || 0}</div>
    <div class="preview-item"><strong>Missing students emailed</strong>${payload.missing_students_receive_email ? "yes" : "no"}</div>
    <div class="preview-item"><strong>Scores</strong>${payload.scores_included ? "included" : "not included"}</div>
    <div class="preview-item"><strong>Attachments</strong>${payload.attachments_included ? "included" : "none"}</div>
    <div class="preview-item"><strong>Subject</strong>${escapeHtml(payload.subject || "")}</div>
  `;
}

function renderReports(reports) {
  const root = $("reportLinks");
  root.innerHTML = "";
  if (!reports.length) {
    root.innerHTML = `<div class="empty-state compact"><p>No reports yet.</p></div>`;
    return;
  }
  for (const report of reports) {
    const link = document.createElement("a");
    link.href = report.url || "#";
    link.textContent = report.name;
    link.target = "_blank";
    root.appendChild(link);
  }
}

async function previewAck({ silent = false } = {}) {
  if (!state.selectedAssignmentId) return;
  const payload = await api(`/api/classes/${state.selectedClassId}/assignments/${state.selectedAssignmentId}/preview-acknowledgements`, { method: "POST" });
  renderReceiptPreview(payload);
  if (!silent) showToast("Preview only. No email sent.");
}

function openConfirmModal(config) {
  state.modalAction = config;
  $("modalEyebrow").textContent = config.eyebrow || "Confirm";
  $("modalTitle").textContent = config.title;
  $("modalBody").innerHTML = config.body;
  $("modalConfirmText").textContent = config.confirmText || "I understand this action.";
  $("modalConfirmCheckbox").checked = false;
  $("modalSendText").value = "";
  $("modalSendTextWrap").hidden = !config.requireSendText;
  $("modalRequiredTextLabel").textContent = config.requiredTextLabel || "Type SEND";
  $("modalConfirmBtn").textContent = config.confirmButton || "Confirm";
  $("confirmModal").hidden = false;
}

function closeModal() {
  $("confirmModal").hidden = true;
  state.modalAction = null;
}

function previewModalBody(preview, { live = false, receipt = false } = {}) {
  const recipients = preview.sample_recipients || preview.recipients || [];
  const recipientLines = recipients.slice(0, 5).map((row) => `${row.student_id || ""} ${row.email || ""}`.trim()).join("\n");
  const sentRecipients = Array.isArray(preview.sent_recipients) ? preview.sent_recipients : [];
  const sentLines = sentRecipients.slice(0, 8).map((row) => `${row.student_id || ""} ${row.email || ""} ${row.sent_at || ""}`.trim()).join("\n");
  return `
    ${
      preview.already_sent
        ? `<div class="modal-preview"><strong>Already sent</strong>\nThis assignment email has already been sent to ${preview.sent_recipient_count || sentRecipients.length} student${(preview.sent_recipient_count || sentRecipients.length) === 1 ? "" : "s"}. Confirming send now will resend it.\n${escapeHtml(sentLines || "No sent recipient details available.")}</div>`
        : ""
    }
    <div class="metric-row">
      <div class="metric"><strong>${preview.recipient_count ?? preview.count ?? 0}</strong><span>recipients</span></div>
      <div class="metric"><strong>${preview.invalid_email_count || 0}</strong><span>invalid emails</span></div>
      <div class="metric"><strong>${escapeHtml(preview.send_at || "pending")}</strong><span>send at</span></div>
      <div class="metric"><strong>${escapeHtml(preview.due_at || "not set")}</strong><span>due at</span></div>
    </div>
    <div class="modal-preview"><strong>Subject</strong>\n${escapeHtml(preview.subject || "")}</div>
    <div class="modal-preview"><strong>Body preview</strong>\n${escapeHtml(preview.body || "")}</div>
    <div class="modal-preview"><strong>Sample recipients</strong>\n${escapeHtml(recipientLines || "None")}</div>
    <p class="helper">${live ? "This will send real email after confirmation." : "This confirms the pending send. No email is sent now."}</p>
    ${receipt ? `<p class="helper">Receipts go only to submitted students. Scores, feedback, and attachments are not included.</p>` : ""}
  `;
}

async function openAssignmentSendLaterModal() {
  const payload = await api(`/api/classes/${state.selectedClassId}/assignments/${state.selectedAssignmentId}/preview-dispatch`, { method: "POST" });
  openConfirmModal({
    title: "Confirm scheduled send",
    body: previewModalBody(payload.preview, { live: false }),
    confirmText: "I confirm this scheduled assignment email. No email will send now.",
    confirmButton: "Confirm send later",
    endpoint: `/api/classes/${state.selectedClassId}/assignments/${state.selectedAssignmentId}/confirm-send-later`,
    payload: { confirm: true },
    success: "Scheduled send confirmed. No email sent."
  });
}

async function openAssignmentSendNowModal() {
  const payload = await api(`/api/classes/${state.selectedClassId}/assignments/${state.selectedAssignmentId}/preview-dispatch`, { method: "POST" });
  const count = Number(payload.preview.recipient_count || 0);
  openConfirmModal({
    title: "Confirm live send",
    body: previewModalBody(payload.preview, { live: true }),
    confirmText: "I confirm this will send real assignment email now.",
    confirmButton: "Confirm send now",
    requireSendText: count > 1,
    endpoint: `/api/classes/${state.selectedClassId}/assignments/${state.selectedAssignmentId}/send-now`,
    payload: { confirm: true },
    success: "Assignment email sent."
  });
}

async function openReceiptLaterModal() {
  const payload = await api(`/api/classes/${state.selectedClassId}/assignments/${state.selectedAssignmentId}/preview-acknowledgements`, { method: "POST" });
  renderReceiptPreview(payload);
  openConfirmModal({
    title: "Confirm scheduled receipts",
    body: previewModalBody(payload, { live: false, receipt: true }),
    confirmText: "I confirm these receipt emails for later. No email will send now.",
    confirmButton: "Confirm send later",
    endpoint: `/api/classes/${state.selectedClassId}/assignments/${state.selectedAssignmentId}/confirm-acknowledgements-later`,
    payload: { confirm: true },
    success: "Receipt send confirmed for later. No email sent."
  });
}

async function openReceiptNowModal() {
  const payload = await api(`/api/classes/${state.selectedClassId}/assignments/${state.selectedAssignmentId}/preview-acknowledgements`, { method: "POST" });
  renderReceiptPreview(payload);
  const count = Number(payload.count || 0);
  openConfirmModal({
    title: "Confirm receipt send",
    body: previewModalBody(payload, { live: true, receipt: true }),
    confirmText: "I confirm this will send real receipt email now.",
    confirmButton: "Confirm send now",
    requireSendText: count > 1,
    endpoint: `/api/classes/${state.selectedClassId}/assignments/${state.selectedAssignmentId}/send-acknowledgements`,
    payload: { confirm: true },
    success: "Receipt email sent."
  });
}

async function confirmModalAction() {
  if (!state.modalAction) return;
  if (!$("modalConfirmCheckbox").checked) {
    showToast("Confirmation is required.");
    return;
  }
  const action = state.modalAction;
  const payload = { ...action.payload };
  if (action.requireSendText) {
    payload.confirm_text = $("modalSendText").value;
  }
  const result = await api(action.endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  closeModal();
  showToast(action.success || result.message || result);
  if (action.afterConfirm === "class_deleted") {
    state.selectedClassId = "";
    state.selectedAssignmentId = "";
    state.classDetail = null;
    localStorage.removeItem(STORAGE_KEYS.selectedClass);
    await loadClasses();
    return;
  }
  if (state.selectedAssignmentId) await openAssignment(state.selectedAssignmentId, state.activeTab);
}

async function openRosterForSelectedClass() {
  if (!state.selectedClassId) return;
  setActiveTab("roster");
  await editRoster(state.selectedClassId);
}

function openDeleteClassModal() {
  if (!state.selectedClassId || !state.classDetail) return;
  const classId = state.selectedClassId;
  const displayName = state.classDetail.class.display_name || classId;
  openConfirmModal({
    eyebrow: "Delete class",
    title: "Delete class",
    body: `
      <div class="modal-preview">This deletes the local class workspace for ${escapeHtml(displayName)} (${escapeHtml(classId)}).</div>
      <p class="helper">This does not send email. Existing external mail is not affected.</p>
    `,
    confirmText: "I understand this will delete the local class workspace.",
    confirmButton: "Delete class",
    requireSendText: true,
    requiredTextLabel: `Type ${classId}`,
    endpoint: `/api/classes/${encodeURIComponent(classId)}/delete`,
    payload: { confirm: true },
    success: "Class deleted.",
    afterConfirm: "class_deleted"
  });
}

async function uploadSubmissions(event) {
  const files = Array.from(event.target.files || []);
  if (!files.length || !state.selectedAssignmentId) return;
  const form = new FormData();
  files.forEach((file) => form.append("submissions", file));
  await api(`/api/classes/${state.selectedClassId}/assignments/${state.selectedAssignmentId}/upload-submissions`, {
    method: "POST",
    body: form
  });
  showToast("Submission PDFs uploaded.");
  event.target.value = "";
}

async function ingestSubmissions(fromMailapp = false) {
  if (!state.selectedAssignmentId) {
    showToast("Open an assignment first.");
    return;
  }
  await api(`/api/classes/${state.selectedClassId}/assignments/${state.selectedAssignmentId}/ingest-submissions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ from_mailapp: fromMailapp, mail_query: state.selectedAssignmentId })
  });
  showToast("Submissions ingested.");
  await openClass(state.selectedClassId, { preserveTab: true });
  await openAssignment(state.selectedAssignmentId, state.activeTab);
}

document.addEventListener("click", async (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;
  const classNavItem = target.closest("[data-open-class]");
  if (target.dataset.hide) hide(target.dataset.hide);
  if (target.dataset.show) show(target.dataset.show);
  if (target.dataset.tab) {
    setActiveTab(target.dataset.tab);
    if (target.dataset.tab === "roster" && state.selectedClassId) await editRoster(state.selectedClassId);
  }
  if (classNavItem instanceof HTMLElement && classNavItem.dataset.openClass) await openClass(classNavItem.dataset.openClass);
  if (target.dataset.openAssignment) await openAssignment(target.dataset.openAssignment, "email");
  if (target.dataset.assignmentTab) await openAssignment(target.dataset.assignmentTab, target.dataset.tabTarget || "email");
  if (target.dataset.removeRow !== undefined) {
    const tr = target.closest("tr");
    tr.querySelector('input[name="active"]').value = "false";
  }
});

$("refreshBtn").addEventListener("click", loadClasses);
$("refreshReportsBtn").addEventListener("click", () => state.selectedAssignmentId && openAssignment(state.selectedAssignmentId, "reports"));
$("showAddClassBtn").addEventListener("click", openAddClassModal);
$("editClassRosterBtn").addEventListener("click", openRosterForSelectedClass);
$("deleteClassBtn").addEventListener("click", openDeleteClassModal);
$("classForm").addEventListener("submit", createClass);
$("toggleRosterBtn").addEventListener("click", async () => {
  if ($("rosterEditor").hidden) await editRoster();
  else {
    $("rosterEditor").hidden = true;
    $("toggleRosterBtn").textContent = "Open roster";
  }
});
$("addStudentBtn").addEventListener("click", () => addRosterRow());
$("saveRosterBtn").addEventListener("click", saveRoster);
$("exportRosterBtn").addEventListener("click", () => window.open(`/api/classes/${state.selectedClassId}/roster/export`, "_blank"));
$("importRosterInput").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const form = new FormData();
  form.append("roster", file);
  await api(`/api/classes/${state.selectedClassId}/roster/import`, { method: "POST", body: form });
  showToast("Roster imported.");
  await editRoster();
});
$("showAddAssignmentBtn").addEventListener("click", () => {
  setActiveTab("assignments");
  show("assignmentFormPanel");
});
$("assignmentForm").addEventListener("submit", createAssignment);
$("sendLaterBtn").addEventListener("click", openAssignmentSendLaterModal);
$("sendNowBtn").addEventListener("click", openAssignmentSendNowModal);
$("receiptLaterBtn").addEventListener("click", openReceiptLaterModal);
$("receiptNowBtn").addEventListener("click", openReceiptNowModal);
$("submissionFiles").addEventListener("change", uploadSubmissions);
$("ingestBtn").addEventListener("click", () => ingestSubmissions(false));
$("ingestMailBtn").addEventListener("click", () => ingestSubmissions(true));
$("modalCloseBtn").addEventListener("click", closeModal);
$("modalCancelBtn").addEventListener("click", closeModal);
$("modalConfirmBtn").addEventListener("click", confirmModalAction);

loadClasses().catch((error) => showToast(error.message));
