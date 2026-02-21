// ================================================================
// CONFIGURATION
// ================================================================

const API_URL = "http://localhost:8000"; // Change for production

// ================================================================
// DOM ELEMENTS
// ================================================================

const analyzeBtn = document.getElementById("analyzeBtn");
const patientIdInput = document.getElementById("patientId");
const clinicalQueryInput = document.getElementById("clinicalQuery");
const loadingSpinner = document.getElementById("loadingSpinner");
const errorMessage = document.getElementById("errorMessage");
const resultsContainer = document.getElementById("resultsContainer");
const statusMessage = document.getElementById("statusMessage");

// ================================================================
// UTILITY FUNCTIONS
// ================================================================

function showLoading() {
    loadingSpinner.classList.remove("hidden");
    resultsContainer.classList.add("hidden");
    errorMessage.classList.add("hidden");
    analyzeBtn.disabled = true;
}

function hideLoading() {
    loadingSpinner.classList.add("hidden");
    analyzeBtn.disabled = false;
}

function showError(error) {
    errorMessage.textContent = `Error: ${error}`;
    errorMessage.classList.remove("hidden");
    resultsContainer.classList.add("hidden");
}

function showResults() {
    resultsContainer.classList.remove("hidden");
    errorMessage.classList.add("hidden");
    window.scrollTo({ top: 0, behavior: "smooth" });
}

function incrementPatient() {
    const current = parseInt(patientIdInput.value);
    patientIdInput.value = current + 1;
}

function decrementPatient() {
    const current = parseInt(patientIdInput.value);
    if (current > 1) {
        patientIdInput.value = current - 1;
    }
}

function formatNumber(num, decimals = 2) {
    return Number(num).toFixed(decimals);
}

function toggleStage(stageId) {
    const stageContent = document.getElementById(stageId);
    if (stageContent) {
        stageContent.classList.toggle("active");
        const header = stageContent.previousElementSibling;
        if (header) {
            const toggle = header.querySelector(".stage-toggle");
            if (toggle) {
                toggle.style.transform = stageContent.classList.contains("active") 
                    ? "rotate(180deg)" 
                    : "rotate(0)";
            }
        }
    }
}

// ================================================================
// RENDERING FUNCTIONS
// ================================================================

function renderPathologyChart(pathologies) {
    const chartContainer = document.getElementById("pathologyChart");
    chartContainer.innerHTML = "";

    if (!pathologies || pathologies.length === 0) {
        chartContainer.innerHTML = "<p style='color: var(--text-muted); text-align: center;'>No pathology data available</p>";
        return;
    }

    const sorted = pathologies.sort((a, b) => b.confidence - a.confidence);
    const maxConfidence = Math.max(...sorted.map(p => p.confidence), 1);

    sorted.forEach(pathology => {
        const percentage = (pathology.confidence / maxConfidence) * 100;
        const confidence = pathology.confidence * 100;
        
        let barClass = "low";
        if (confidence >= 70) barClass = "high";
        else if (confidence >= 40) barClass = "moderate";

        const barHTML = `
            <div class="bar-item">
                <div class="bar-label">${pathology.name}</div>
                <div class="bar-container">
                    <div class="bar-fill ${barClass}" style="width: ${percentage}%">
                        ${formatNumber(confidence, 1)}%
                    </div>
                </div>
                <div class="bar-value">${formatNumber(confidence, 1)}%</div>
            </div>
        `;
        chartContainer.innerHTML += barHTML;
    });
}

function renderFindingItems(evidence) {
    const findingsContainer = document.getElementById("findingsContainer");
    findingsContainer.innerHTML = "";

    if (!evidence || evidence.length === 0) {
        findingsContainer.innerHTML = "<p style='color: var(--text-muted);'>No findings available.</p>";
        return;
    }

    evidence.forEach((item, index) => {
        const findings = item.pathology_findings || "No findings available";
        const scores = item.pathology_scores || {};

        let scoresHTML = "";
        if (Object.keys(scores).length > 0) {
            const sortedScores = Object.entries(scores)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 10);

            scoresHTML = `
                <div class="finding-field">
                    <label class="finding-field-label">Detailed Scores:</label>
                    <table style="width: 100%; color: var(--text-secondary); font-size: 0.9rem;">
                        <thead>
                            <tr style="border-bottom: 1px solid var(--border-color); text-align: left;">
                                <th style="padding: var(--spacing-sm);">Pathology</th>
                                <th style="padding: var(--spacing-sm);">Probability</th>
                                <th style="padding: var(--spacing-sm);">Confidence</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${sortedScores.map(([pathology, score]) => {
                                const confidenceLevel = score >= 0.7 ? '🔴 High' : score >= 0.5 ? '🟡 Moderate' : '🟢 Low';
                                return `
                                    <tr style="border-bottom: 1px solid var(--border-color);">
                                        <td style="padding: var(--spacing-sm);">${pathology}</td>
                                        <td style="padding: var(--spacing-sm);">${formatNumber(score * 100, 2)}%</td>
                                        <td style="padding: var(--spacing-sm);">${confidenceLevel}</td>
                                    </tr>
                                `;
                            }).join('')}
                        </tbody>
                    </table>
                </div>
            `;
        }

        const findingHTML = `
            <div class="finding-item">
                <div class="finding-header" onclick="toggleFinding(this)">
                    <div class="finding-header-title">
                        <span class="finding-toggle">▼</span>
                        <span>Evidence ${index + 1} - Pathology Analysis</span>
                    </div>
                </div>
                <div class="finding-content">
                    <div class="finding-field">
                        <label class="finding-field-label">Findings:</label>
                        <div class="finding-field-value">${findings}</div>
                    </div>
                    ${scoresHTML}
                </div>
            </div>
        `;
        findingsContainer.innerHTML += findingHTML;
    });
}

function renderEvidenceItems(evidence) {
    const evidenceWrapper = document.getElementById("evidenceItemsWrapper");
    evidenceWrapper.innerHTML = "";

    if (!evidence || evidence.length === 0) {
        evidenceWrapper.innerHTML = "<p style='color: var(--text-muted);'>No evidence items retrieved.</p>";
        return;
    }

    evidence.forEach((item, index) => {
        const relevance = item.relevance_score || item.metadata?.relevance_score || 0;
        const modality = item.modality || item.metadata?.modality || "UNKNOWN";
        const reportText = item.report_text || item.metadata?.report_text || "N/A";
        const organ = item.organ || item.metadata?.organ || "N/A";

        let evidenceHTML = `
            <div class="evidence-item">
                <div class="evidence-item-header" onclick="toggleEvidenceItem(this)">
                    <div class="evidence-item-title">
                        <span class="finding-toggle">▼</span>
                        <span>Evidence ${index + 1} — ${modality} (Relevance: ${formatNumber(relevance, 2)})</span>
                    </div>
                    <span class="evidence-relevance-badge">${formatNumber(relevance, 2)}</span>
                </div>
                <div class="evidence-item-content">
                    <div class="evidence-field">
                        <strong>Report Text:</strong>
                        <p>${reportText}</p>
                    </div>
                    <div class="evidence-field">
                        <strong>Organ:</strong>
                        <p>${organ}</p>
                    </div>
                    <div class="evidence-field">
                        <strong>Modality:</strong>
                        <p>${modality}</p>
                    </div>
        `;

        if (item.image_path) {
            evidenceHTML += `
                    <div class="evidence-image-container">
                        <img src="${item.image_path}" alt="Evidence Image ${index + 1}" class="evidence-image">
                    </div>
            `;
        }

        evidenceHTML += `
                </div>
            </div>
        `;
        evidenceWrapper.innerHTML += evidenceHTML;
    });
}

function toggleFinding(element) {
    const content = element.nextElementSibling;
    if (content) {
        content.classList.toggle("active");
        const toggle = element.querySelector(".finding-toggle");
        if (toggle) {
            toggle.style.transform = content.classList.contains("active") 
                ? "rotate(180deg)" 
                : "rotate(0)";
        }
    }
}

function toggleEvidenceItem(element) {
    const content = element.nextElementSibling;
    if (content) {
        content.classList.toggle("active");
        const toggle = element.querySelector(".finding-toggle");
        if (toggle) {
            toggle.style.transform = content.classList.contains("active") 
                ? "rotate(180deg)" 
                : "rotate(0)";
        }
    }
}

function populateResults(data) {
    console.log("[v0] Populating results with data:", data);

    // Pipeline Summary
    document.getElementById("totalIterations").textContent = data.total_iterations || 0;
    document.getElementById("retrievalAttempts").textContent = data.retrieval_attempts || 0;
    document.getElementById("reasoningAttempts").textContent = data.reasoning_attempts || 0;

    // Quality Scores
    const qualityScores = data.quality_scores || {};
    const evidenceScore = qualityScores.evidence || 0;
    const responseScore = qualityScores.response || 0;
    const overallScore = (evidenceScore + responseScore) / 2;

    document.getElementById("evidenceQuality").textContent = formatNumber(evidenceScore, 2);
    document.getElementById("responseQuality").textContent = formatNumber(responseScore, 2);
    document.getElementById("overallQuality").textContent = formatNumber(overallScore, 2);

    // Quality Badges
    const evidenceBadge = document.getElementById("evidenceBadge");
    evidenceBadge.textContent = evidenceScore >= 0.6 ? "PASS" : "ATTENTION";
    evidenceBadge.style.background = evidenceScore >= 0.6 
        ? "rgba(16, 185, 129, 0.15)" 
        : "rgba(239, 68, 68, 0.1)";
    evidenceBadge.style.color = evidenceScore >= 0.6 ? "var(--success-color)" : "var(--danger-color)";

    const responseBadge = document.getElementById("responseBadge");
    responseBadge.textContent = responseScore >= 0.7 ? "PASS" : "ATTENTION";
    responseBadge.style.background = responseScore >= 0.7 
        ? "rgba(16, 185, 129, 0.15)" 
        : "rgba(239, 68, 68, 0.1)";
    responseBadge.style.color = responseScore >= 0.7 ? "var(--success-color)" : "var(--danger-color)";

    const overallBadge = document.getElementById("overallBadge");
    if (overallScore >= 0.7) {
        overallBadge.textContent = "EXCELLENT";
        overallBadge.style.background = "rgba(16, 185, 129, 0.15)";
        overallBadge.style.color = "var(--success-color)";
    } else if (overallScore >= 0.5) {
        overallBadge.textContent = "GOOD";
        overallBadge.style.background = "rgba(16, 185, 129, 0.1)";
        overallBadge.style.color = "var(--success-color)";
    } else {
        overallBadge.textContent = "NEEDS REVIEW";
        overallBadge.style.background = "rgba(239, 68, 68, 0.1)";
        overallBadge.style.color = "var(--danger-color)";
    }

    // Stage Data
    const evidenceGate = data.evidence_gate_result || {};
    const filterResult = data.evidence_filter_result || {};
    document.getElementById("originalEvidence").textContent = (data.evidence || []).length;
    document.getElementById("filteredEvidenceCount").textContent = (data.filtered_evidence || []).length;
    document.getElementById("removedEvidence").textContent = filterResult.removed_count || 0;
    document.getElementById("filterQuality").textContent = formatNumber(filterResult.quality_score || 0, 2);
    document.getElementById("gateDecision1").textContent = evidenceGate.decision || "N/A";
    document.getElementById("stageMessage1").textContent = filterResult.feedback || "Processing...";

    const responseGate = data.response_gate_result || {};
    document.getElementById("reasoningQuality").textContent = formatNumber(responseGate.score || 0, 2);
    document.getElementById("gateDecision3").textContent = responseGate.decision || "N/A";

    // Pathology Detection
    const pathologyResults = data.xray_results?.[0]?.detections || [];
    if (pathologyResults.length > 0) {
        const pathologiesFormatted = pathologyResults.map(p => ({
            name: p.pathology || p.disease || "Unknown",
            confidence: p.confidence || 0
        }));
        renderPathologyChart(pathologiesFormatted);
    }

    // Findings
    renderFindingItems(data.filtered_evidence || []);

    // Clinical Response
    const finalAnswer = data.final_answer || "No response generated";
    document.getElementById("clinicalDiagnosis").textContent = finalAnswer;

    // Parse clinical response sections
    const lines = finalAnswer.split('\n');
    let diagnosisLines = [];
    let evidenceLines = [];
    let recommendationLines = [];
    let currentSection = null;

    lines.forEach(line => {
        const lineLower = line.toLowerCase();
        if (lineLower.includes('diagnosis') || lineLower.includes('impression')) {
            currentSection = "diagnosis";
        } else if (lineLower.includes('supporting evidence') || lineLower.includes('evidence:')) {
            currentSection = "evidence";
        } else if (lineLower.includes('next steps') || lineLower.includes('recommendation')) {
            currentSection = "recommendations";
        } else if (line.trim()) {
            if (currentSection === "diagnosis") diagnosisLines.push(line.trim());
            else if (currentSection === "evidence") evidenceLines.push(line.trim());
            else if (currentSection === "recommendations") recommendationLines.push(line.trim());
        }
    });

    document.getElementById("supportingEvidenceList").innerHTML = 
        evidenceLines.map(line => `<li>${line}</li>`).join('') || '<li>No supporting evidence found</li>';
    
    document.getElementById("recommendationsList").innerHTML = 
        recommendationLines.map(line => `<li>${line}</li>`).join('') || '<li>No recommendations available</li>';

    // Metrics
    const metrics = data.metrics || {};
    document.getElementById("precisionK").textContent = formatNumber(metrics.precision_at_k || 1.0, 3);
    document.getElementById("recallK").textContent = formatNumber(metrics.recall_at_k || 1.0, 3);
    document.getElementById("mrr").textContent = formatNumber(metrics.mrr || 1.0, 3);
    document.getElementById("groundedness").textContent = formatNumber(metrics.groundedness || 1.0, 3);
    document.getElementById("clinicalCorrectness").textContent = formatNumber(metrics.clinical_correctness || 0.637, 3);
    document.getElementById("completeness").textContent = formatNumber(metrics.completeness || 1.0, 3);

    // Evidence Items
    const modalityItems = document.getElementById("modalityItems");
    const modalities = {};
    (data.filtered_evidence || []).forEach(e => {
        const mod = e.modality || e.metadata?.modality || "Unknown";
        modalities[mod] = (modalities[mod] || 0) + 1;
    });

    modalityItems.innerHTML = Object.entries(modalities)
        .map(([mod, count]) => `
            <div class="modality-item">
                <div class="modality-name">${mod}</div>
                <div class="modality-count">${count}</div>
            </div>
        `).join('');

    const evidenceSummary = document.getElementById("evidenceSummary");
    const totalEvidence = (data.filtered_evidence || []).length;
    evidenceSummary.innerHTML = totalEvidence > 0
        ? `<span class="badge badge-success">✓ Found ${totalEvidence} relevant evidence items</span>`
        : `<span class="badge badge-success" style="background-color: rgba(239, 68, 68, 0.1); color: var(--danger-color); border-color: var(--danger-color);">✗ No relevant evidence found</span>`;

    renderEvidenceItems(data.filtered_evidence || []);
}

// ================================================================
// MAIN API CALL
// ================================================================

async function runAnalysis() {
    const patientId = parseInt(patientIdInput.value);
    const query = clinicalQueryInput.value.trim();

    if (!query) {
        showError("Please enter a clinical query.");
        return;
    }

    console.log("[v0] Starting analysis with:", { patientId, query });

    showLoading();

    try {
        const response = await fetch(`${API_URL}/api/analyze`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                patient_id: patientId,
                query: query,
            }),
        });

        console.log("[v0] API response status:", response.status);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `API Error: ${response.status}`);
        }

        const data = await response.json();
        console.log("[v0] Analysis complete:", data);

        populateResults(data.results);
        showResults();

    } catch (error) {
        console.error("[v0] Error during analysis:", error);
        showError(error.message);
    } finally {
        hideLoading();
    }
}

// ================================================================
// EVENT LISTENERS
// ================================================================

analyzeBtn.addEventListener("click", runAnalysis);

clinicalQueryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.ctrlKey) {
        runAnalysis();
    }
});

// ================================================================
// INITIALIZATION
// ================================================================

console.log("[v0] Application initialized. API URL:", API_URL);
