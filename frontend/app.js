// GlycoGuard AI Frontend Logic
const API_BASE = window.location.origin;

let modelMetadata = null;
let whatifChartInstance = null;
let importanceChartInstance = null;
let currentBatchResults = [];

// Clinical Presets
const PRESETS = {
  healthy: {
    Pregnancies: 1,
    Glucose: 85,
    BloodPressure: 68,
    SkinThickness: 20,
    Insulin: 65,
    BMI: 22.4,
    DiabetesPedigreeFunction: 0.25,
    Age: 26
  },
  prediabetic: {
    Pregnancies: 3,
    Glucose: 128,
    BloodPressure: 78,
    SkinThickness: 28,
    Insulin: 110,
    BMI: 28.7,
    DiabetesPedigreeFunction: 0.48,
    Age: 44
  },
  highrisk: {
    Pregnancies: 6,
    Glucose: 175,
    BloodPressure: 88,
    SkinThickness: 38,
    Insulin: 210,
    BMI: 36.8,
    DiabetesPedigreeFunction: 0.85,
    Age: 52
  },
  gestational: {
    Pregnancies: 8,
    Glucose: 142,
    BloodPressure: 82,
    SkinThickness: 32,
    Insulin: 140,
    BMI: 31.5,
    DiabetesPedigreeFunction: 0.62,
    Age: 38
  }
};

// Parameter Display units
const UNITS = {
  Glucose: 'mg/dL',
  BMI: 'kg/m²',
  BloodPressure: 'mm Hg',
  SkinThickness: 'mm',
  Insulin: 'mu U/ml',
  Age: 'yrs',
  Pregnancies: '',
  DiabetesPedigreeFunction: ''
};

// Navigation
function switchTab(tabId) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('block'));
  
  const target = document.getElementById(tabId);
  if (target) {
    target.classList.remove('hidden');
    target.classList.add('block');
  }

  document.querySelectorAll('.nav-tab').forEach(btn => {
    btn.classList.remove('active-tab');
    btn.classList.add('border-transparent', 'text-slate-600');
  });

  const activeBtn = document.getElementById(`btn-${tabId}`);
  if (activeBtn) {
    activeBtn.classList.add('active-tab');
    activeBtn.classList.remove('border-transparent', 'text-slate-600');
  }

  // Trigger chart resize / re-render if switching to charts
  if (tabId === 'tab-whatif') {
    setTimeout(runWhatIfSimulation, 100);
  } else if (tabId === 'tab-analytics') {
    setTimeout(renderAnalyticsCharts, 100);
  }
}

// Parameter Synchronization
function syncParam(param, value) {
  const label = document.getElementById(`val-${param}`);
  const unit = UNITS[param] || '';
  if (label) {
    label.innerText = `${value} ${unit}`.trim();
  }
}

function getPatientFromInputs() {
  return {
    Pregnancies: parseInt(document.getElementById('range-Pregnancies').value, 10),
    Glucose: parseFloat(document.getElementById('range-Glucose').value),
    BloodPressure: parseFloat(document.getElementById('range-BloodPressure').value),
    SkinThickness: parseFloat(document.getElementById('range-SkinThickness').value),
    Insulin: parseFloat(document.getElementById('range-Insulin').value),
    BMI: parseFloat(document.getElementById('range-BMI').value),
    DiabetesPedigreeFunction: parseFloat(document.getElementById('range-DiabetesPedigreeFunction').value),
    Age: parseInt(document.getElementById('range-Age').value, 10)
  };
}

function setInputsFromPatient(p) {
  for (const [key, val] of Object.entries(p)) {
    const el = document.getElementById(`range-${key}`);
    if (el) {
      el.value = val;
      syncParam(key, val);
    }
  }
}

function loadPreset(name) {
  const preset = PRESETS[name];
  if (preset) {
    setInputsFromPatient(preset);
    runPrediction();
  }
}

function resetInputs() {
  loadPreset('healthy');
}

// Prediction Call
async function runPrediction() {
  const patient = getPatientFromInputs();
  const btn = document.getElementById('btn-assess');
  const originalText = btn.innerHTML;
  btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Evaluating Patient...`;
  btn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patient)
    });

    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    const data = await res.json();
    renderPredictionResult(data);
  } catch (err) {
    console.error("Prediction error:", err);
    alert("Failed to communicate with prediction API. Check console for details.");
  } finally {
    btn.innerHTML = originalText;
    btn.disabled = false;
  }
}

function renderPredictionResult(data) {
  // Update gauge circle
  const circle = document.getElementById('gauge-circle');
  const circumference = 2 * Math.PI * 40; // 251.2
  const offset = circumference * (1 - (data.risk_probability));
  circle.style.strokeDashoffset = offset;
  circle.style.stroke = data.risk_color;

  // Update text elements
  document.getElementById('risk-percentage').innerText = `${data.risk_percentage}%`;
  document.getElementById('risk-percentage').style.color = data.risk_color;

  const badge = document.getElementById('risk-badge');
  badge.innerText = data.risk_tier;
  badge.style.backgroundColor = `${data.risk_color}18`;
  badge.style.color = data.risk_color;
  badge.style.borderColor = `${data.risk_color}40`;

  document.getElementById('outcome-label').innerText = data.outcome_label;
  document.getElementById('confidence-text').innerText = `Model Confidence: ${data.confidence_score}%`;

  // Update Recommendations
  const recList = document.getElementById('recommendations-list');
  recList.innerHTML = '';
  data.clinical_recommendations.forEach(rec => {
    const li = document.createElement('li');
    li.className = 'flex items-start gap-2';
    li.innerHTML = `
      <i class="fa-solid fa-check-circle text-teal-600 mt-0.5"></i>
      <span>${rec}</span>
    `;
    recList.appendChild(li);
  });

  // Update XAI local feature cards
  const xaiContainer = document.getElementById('xai-container');
  xaiContainer.innerHTML = '';
  data.local_explanations.forEach(exp => {
    const isElevated = exp.status === 'elevated';
    const isBelow = exp.status === 'below';
    const statusBg = isElevated ? 'bg-red-50 text-red-700 border-red-200' : 
                     isBelow ? 'bg-blue-50 text-blue-700 border-blue-200' : 'bg-emerald-50 text-emerald-700 border-emerald-200';
    const arrow = isElevated ? '▲' : isBelow ? '▼' : '✔';

    const card = document.createElement('div');
    card.className = `p-3.5 rounded-xl border ${statusBg} flex flex-col justify-between`;
    card.innerHTML = `
      <div class="flex items-center justify-between">
        <span class="text-xs font-bold text-slate-800">${exp.feature}</span>
        <span class="text-[10px] font-bold px-1.5 py-0.5 rounded bg-white/80 border border-current">${arrow} ${Math.abs(exp.deviation_pct)}%</span>
      </div>
      <div class="mt-2 text-xs">
        <div class="flex justify-between text-slate-600">
          <span>Patient Value:</span>
          <span class="font-bold text-slate-900">${exp.patient_value}</span>
        </div>
        <div class="flex justify-between text-slate-500 text-[11px] mt-0.5">
          <span>Healthy Baseline:</span>
          <span>${exp.healthy_baseline}</span>
        </div>
      </div>
      <div class="mt-2 text-[10px] font-semibold text-slate-500 uppercase tracking-wider">
        ${exp.risk_level}
      </div>
    `;
    xaiContainer.appendChild(card);
  });
}

// What-If Simulation
async function runWhatIfSimulation() {
  const patient = getPatientFromInputs();
  const variable = document.getElementById('whatif-variable').value;

  try {
    const res = await fetch(`${API_BASE}/api/what-if`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        base_patient: patient,
        variable_to_sweep: variable,
        steps: 35
      })
    });

    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    const data = await res.json();
    renderWhatIfChart(data);
  } catch (err) {
    console.error("What-if simulation error:", err);
  }
}

function renderWhatIfChart(data) {
  const ctx = document.getElementById('whatif-chart').getContext('2d');
  const labels = data.curve.map(pt => pt.value);
  const values = data.curve.map(pt => pt.risk_percentage);

  // Inflection alert
  const infContainer = document.getElementById('inflection-container');
  if (data.inflection_point) {
    infContainer.innerHTML = `
      <div class="p-3 bg-red-50 border border-red-200 rounded-xl text-red-800">
        <div class="font-bold text-red-900 mb-1">
          <i class="fa-solid fa-triangle-exclamation"></i> Transition at ${data.variable} = ${data.inflection_point.threshold_value}
        </div>
        <p>${data.inflection_point.description}</p>
      </div>
    `;
  } else {
    infContainer.innerHTML = `
      <div class="p-3 bg-emerald-50 border border-emerald-200 rounded-xl text-emerald-800">
        <div class="font-bold text-emerald-900 mb-1">
          <i class="fa-solid fa-circle-check"></i> Stable Physiological Zone
        </div>
        <p>Across this range, the patient's predicted risk remained within the non-diabetic safe threshold.</p>
      </div>
    `;
  }

  document.getElementById('whatif-summary').innerText = data.summary;

  if (whatifChartInstance) {
    whatifChartInstance.destroy();
  }

  whatifChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: `${data.variable} Sweep Risk %`,
          data: values,
          borderColor: '#0d9488',
          backgroundColor: 'rgba(13, 148, 136, 0.1)',
          fill: true,
          tension: 0.3,
          borderWidth: 3,
          pointRadius: 2,
          pointHoverRadius: 6
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `Risk: ${ctx.parsed.y}%`
          }
        }
      },
      scales: {
        y: {
          min: 0,
          max: 100,
          title: { display: true, text: 'Predicted Risk Probability (%)', font: { size: 11, weight: 'bold' } },
          grid: { color: '#f1f5f9' }
        },
        x: {
          title: { display: true, text: `${data.variable} Value`, font: { size: 11, weight: 'bold' } },
          grid: { color: '#f1f5f9' }
        }
      }
    }
  });
}

// Batch CSV Screening
async function handleCSVUpload(input) {
  const file = input.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(`${API_BASE}/api/predict/batch`, {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    currentBatchResults = data.results;
    renderBatchResults(data);
  } catch (err) {
    alert(`Batch screening error: ${err.message}`);
  }
}

function renderBatchResults(data) {
  document.getElementById('batch-results-section').classList.remove('hidden');
  document.getElementById('batch-total').innerText = data.total_processed;
  document.getElementById('batch-diabetic').innerText = data.diabetic_count;
  document.getElementById('batch-nondiabetic').innerText = data.non_diabetic_count;
  document.getElementById('batch-avg-risk').innerText = `${data.average_risk_percentage}%`;

  const tbody = document.getElementById('batch-table-body');
  tbody.innerHTML = '';

  data.results.slice(0, 50).forEach((row, idx) => {
    const tr = document.createElement('tr');
    const isDiabetic = row.Prediction === 1;
    const badgeColor = isDiabetic ? 'bg-red-50 text-red-700 border-red-200' : 'bg-emerald-50 text-emerald-700 border-emerald-200';

    tr.innerHTML = `
      <td class="p-3 font-semibold text-slate-500">${idx + 1}</td>
      <td class="p-3 font-medium text-slate-900">${row.Glucose}</td>
      <td class="p-3">${row.BMI}</td>
      <td class="p-3">${row.BloodPressure}</td>
      <td class="p-3">${row.Age}</td>
      <td class="p-3 font-bold">${row.Risk_Percentage}%</td>
      <td class="p-3">
        <span class="px-2 py-0.5 rounded text-[11px] font-bold border ${badgeColor}">
          ${isDiabetic ? 'Diabetic Risk' : 'Non-Diabetic'}
        </span>
      </td>
      <td class="p-3 text-[11px] text-slate-600">${row.Risk_Tier}</td>
    `;
    tbody.appendChild(tr);
  });
}

function downloadSampleCSV() {
  const csvContent = 
`Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
1,85,66,29,0,26.6,0.351,31
8,183,64,0,0,23.3,0.672,32
1,89,66,23,94,28.1,0.167,21
0,137,40,35,168,43.1,2.288,33
5,116,74,0,0,25.6,0.201,30
3,78,50,32,88,31.0,0.248,26
10,115,0,0,0,35.3,0.134,29
2,197,70,45,543,30.5,0.158,53
1,189,60,23,846,30.1,0.398,59
5,166,72,19,175,25.8,0.587,51`;

  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = "sample_patients_cohort.csv";
  a.click();
  URL.revokeObjectURL(url);
}

function exportBatchResults() {
  if (!currentBatchResults.length) return;
  const headers = Object.keys(currentBatchResults[0]);
  const rows = currentBatchResults.map(r => headers.map(h => `"${r[h]}"`).join(','));
  const csvContent = [headers.join(','), ...rows].join('\n');

  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = "glycoguard_scored_cohort.csv";
  a.click();
  URL.revokeObjectURL(url);
}

// Load Model Info & Analytics
async function fetchModelInfo() {
  try {
    const res = await fetch(`${API_BASE}/api/model-info`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    modelMetadata = await res.json();

    document.getElementById('header-roc-auc').innerText = modelMetadata.cv_roc_auc.toFixed(4);
    document.getElementById('metric-cv-auc').innerText = modelMetadata.cv_roc_auc.toFixed(4);
    document.getElementById('metric-recall').innerText = modelMetadata.test_metrics.recall.toFixed(4);
    document.getElementById('metric-spec').innerText = modelMetadata.test_metrics.specificity.toFixed(4);
    document.getElementById('metric-f1').innerText = modelMetadata.test_metrics.f1_score.toFixed(4);

    const cm = modelMetadata.test_metrics.confusion_matrix;
    document.getElementById('cm-tn').innerText = cm.true_negatives;
    document.getElementById('cm-fp').innerText = cm.false_positives;
    document.getElementById('cm-fn').innerText = cm.false_negatives;
    document.getElementById('cm-tp').innerText = cm.true_positives;

    renderBenchmarkTable(modelMetadata.benchmarks);
  } catch (err) {
    console.error("Failed to load model metadata:", err);
  }
}

function renderBenchmarkTable(benchmarks) {
  const tbody = document.getElementById('benchmark-table-body');
  if (!tbody) return;
  tbody.innerHTML = '';

  benchmarks.forEach(b => {
    const tr = document.createElement('tr');
    const isSelected = b.model.includes('Random Forest');
    if (isSelected) tr.className = 'bg-teal-50/60 font-semibold';

    tr.innerHTML = `
      <td class="p-3 font-medium text-slate-900 flex items-center gap-1.5">
        ${isSelected ? '<i class="fa-solid fa-crown text-amber-500"></i>' : ''}
        ${b.model}
      </td>
      <td class="p-3 font-bold text-teal-700">${b.cv_roc_auc.toFixed(4)}</td>
      <td class="p-3">${b.cv_recall.toFixed(4)}</td>
      <td class="p-3">${b.cv_specificity.toFixed(4)}</td>
      <td class="p-3">${b.cv_f1.toFixed(4)}</td>
      <td class="p-3">${b.cv_accuracy.toFixed(4)}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderAnalyticsCharts() {
  if (!modelMetadata || !modelMetadata.feature_importances) return;

  const ctx = document.getElementById('importance-chart').getContext('2d');
  const feats = modelMetadata.feature_importances.slice(0, 10);
  const labels = feats.map(f => f.feature);
  const values = feats.map(f => f.importance);

  if (importanceChartInstance) {
    importanceChartInstance.destroy();
  }

  importanceChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Relative Importance Weight',
        data: values,
        backgroundColor: '#0d9488',
        borderRadius: 6
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          title: { display: true, text: 'Gini Importance Score', font: { size: 11, weight: 'bold' } },
          grid: { color: '#f1f5f9' }
        },
        y: {
          grid: { display: false }
        }
      }
    }
  });
}

// Initial Boot
document.addEventListener('DOMContentLoaded', () => {
  fetchModelInfo();
  loadPreset('prediabetic');
});
