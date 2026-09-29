// app/frontend/static/js/app.js

document.addEventListener('DOMContentLoaded', () => {
    initDashboard();
});

let mainChart = null;
let spreadChart = null;
let trajectoryChart = null;

let currentProductFilter = 'MG95';
let currentTrajProduct = 'MG95';
let currentHorizon = 7;

let historicalDataCache = [];
let crackSpreadCache = [];
let vietnamForecastCache = null;
let multiHorizonCache = null;

async function initDashboard() {
    setupEventListeners();
    await Promise.all([
        loadMarketOverview(),
        loadMultiHorizonForecast(currentHorizon),
        loadHistoricalChart(),
        loadCrackSpreadChart(),
        loadMetricsTable(),
        loadVietnamForecast()
    ]);
}

function setupEventListeners() {
    // 1. Live forecast modal button
    const btnForecast = document.getElementById('btnLiveForecast');
    if (btnForecast) {
        btnForecast.addEventListener('click', showLiveForecastModal);
    }

    // Modal close
    const btnCloseModal = document.getElementById('btnCloseModal');
    const modalBackdrop = document.getElementById('forecastModal');
    if (btnCloseModal && modalBackdrop) {
        btnCloseModal.addEventListener('click', () => modalBackdrop.classList.remove('active'));
        modalBackdrop.addEventListener('click', (e) => {
            if (e.target === modalBackdrop) modalBackdrop.classList.remove('active');
        });
    }

    // 2. Market Mode Switcher (Singapore vs Vietnam)
    const btnModeSG = document.getElementById('btnModeSG');
    const btnModeVN = document.getElementById('btnModeVN');
    const secSG = document.getElementById('sectionSingaporeKPI');
    const secVN = document.getElementById('sectionVietnamKPI');

    if (btnModeSG && btnModeVN && secSG && secVN) {
        btnModeSG.addEventListener('click', () => {
            btnModeSG.classList.add('active');
            btnModeVN.classList.remove('active-vn', 'active');
            secSG.style.display = 'block';
            secVN.style.display = 'none';
        });

        btnModeVN.addEventListener('click', () => {
            btnModeVN.classList.add('active-vn');
            btnModeSG.classList.remove('active');
            secSG.style.display = 'none';
            secVN.style.display = 'block';
            loadVietnamForecast();
        });
    }

    // 3. Multi-Horizon Selector Buttons (1, 3, 7, 20 Days)
    document.querySelectorAll('.horizon-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.horizon-btn').forEach(b => b.classList.remove('active'));
            const target = e.currentTarget;
            target.classList.add('active');
            currentHorizon = parseInt(target.getAttribute('data-horizon')) || 7;
            
            // Reload multi-horizon & Vietnam forecasts
            loadMultiHorizonForecast(currentHorizon);
            loadVietnamForecast();
        });
    });

    // 4. Trajectory Product Filters
    document.querySelectorAll('.btn-traj-filter').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.btn-traj-filter').forEach(b => b.classList.remove('active'));
            e.currentTarget.classList.add('active');
            currentTrajProduct = e.currentTarget.getAttribute('data-product');
            renderTrajectoryChart();
        });
    });

    // 5. Tax Modal close
    const btnCloseTaxModal = document.getElementById('btnCloseTaxModal');
    const taxModal = document.getElementById('taxModal');
    if (btnCloseTaxModal && taxModal) {
        btnCloseTaxModal.addEventListener('click', () => taxModal.classList.remove('active'));
        taxModal.addEventListener('click', (e) => {
            if (e.target === taxModal) taxModal.classList.remove('active');
        });
    }

    // 6. Update Vietnam FX
    const btnUpdateVnFx = document.getElementById('btnUpdateVnFx');
    if (btnUpdateVnFx) {
        btnUpdateVnFx.addEventListener('click', () => {
            const fx = parseFloat(document.getElementById('inputVnFx').value) || 25400;
            loadVietnamForecast(fx);
        });
    }

    // 7. Historical Chart product filter buttons
    document.querySelectorAll('.btn-filter[data-product]').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.btn-filter[data-product]').forEach(b => b.classList.remove('active'));
            e.currentTarget.classList.add('active');
            currentProductFilter = e.currentTarget.getAttribute('data-product');
            updateMainChartSeries();
        });
    });

    // 8. Simulation Sliders
    const gasSlider = document.getElementById('gasShockSlider');
    const dieselSlider = document.getElementById('dieselShockSlider');
    const volSlider = document.getElementById('volSlider');

    const gasVal = document.getElementById('gasShockVal');
    const dieselVal = document.getElementById('dieselShockVal');
    const volVal = document.getElementById('volVal');

    if (gasSlider && gasVal) {
        gasSlider.addEventListener('input', (e) => {
            const v = parseFloat(e.target.value);
            gasVal.textContent = (v > 0 ? '+' : '') + v.toFixed(1) + '%';
        });
    }

    if (dieselSlider && dieselVal) {
        dieselSlider.addEventListener('input', (e) => {
            const v = parseFloat(e.target.value);
            dieselVal.textContent = (v > 0 ? '+' : '') + v.toFixed(1) + '%';
        });
    }

    if (volSlider && volVal) {
        volSlider.addEventListener('input', (e) => {
            const v = parseFloat(e.target.value);
            volVal.textContent = 'x' + v.toFixed(1);
        });
    }

    // Run simulation button
    const btnSimulate = document.getElementById('btnSimulate');
    if (btnSimulate) {
        btnSimulate.addEventListener('click', runStressSimulation);
    }
}

// 1. Load Market Overview & Baseline Prices
async function loadMarketOverview() {
    try {
        const res = await fetch('/api/overview');
        const data = await res.json();

        // Update header info
        const dateEl = document.getElementById('headerLastDate');
        if (dateEl) dateEl.textContent = data.last_updated;

        const hwEl = document.getElementById('hardwareTag');
        if (hwEl) hwEl.textContent = data.hardware;

        // Render current prices and sparklines
        const products = ['MG95', 'MG92', 'DO_0001', 'DO_005'];
        products.forEach(p => {
            const pData = data.products[p];
            if (!pData) return;

            const card = document.getElementById(`card-${p}`);
            if (!card) return;

            card.querySelector('.price-main').textContent = `${pData.current_price.toFixed(2)}$`;

            // Draw sparkline canvas
            const canvas = card.querySelector('.sparkline-canvas');
            if (canvas && pData.sparkline) {
                drawSparkline(canvas, pData.sparkline, pData.delta_predicted >= 0 ? '#10b981' : '#f43f5e');
            }
        });
    } catch (err) {
        console.error("Lỗi tải Market Overview:", err);
    }
}

// 2. Load Multi-Horizon Forecast (T+1, T+3, T+7, T+20)
async function loadMultiHorizonForecast(horizon = 7) {
    try {
        const res = await fetch(`/api/forecast/multi-horizon?horizon=${horizon}`);
        const data = await res.json();
        multiHorizonCache = data;

        // Update Horizon Bar Info
        const targetDateEl = document.getElementById('hzTargetDate');
        if (targetDateEl) targetDateEl.textContent = data.target_date;

        const contextDescEl = document.getElementById('hzContextDesc');
        if (contextDescEl) contextDescEl.textContent = `🎯 ${data.business_context}`;

        // Update Singapore KPI Cards
        data.predictions.forEach(p => {
            const card = document.getElementById(`card-${p.product}`);
            if (!card) return;

            // Update label
            const labelEl = card.querySelector('.forecast-label');
            if (labelEl) labelEl.textContent = `Dự báo T+${data.horizon}:`;

            // Update forecasted value
            const valEl = card.querySelector('.forecast-val');
            if (valEl) valEl.textContent = `${p.predicted_price.toFixed(2)}$`;

            // Update delta pill
            const deltaPill = card.querySelector('.delta-pill');
            if (deltaPill) {
                const sign = p.delta >= 0 ? '+' : '';
                deltaPill.textContent = `${sign}${p.delta.toFixed(2)}$ (${sign}${p.delta_pct.toFixed(2)}%)`;
                deltaPill.className = `delta-pill ${p.delta >= 0 ? 'delta-pos' : 'delta-neg'}`;
            }

            // Update trend badge
            const trendBadge = card.querySelector('.trend-badge');
            if (trendBadge) {
                const isPos = p.delta >= 0;
                trendBadge.textContent = isPos ? `▲ ${p.signal}` : `▼ ${p.signal}`;
                trendBadge.className = `trend-badge ${isPos ? 'trend-bullish' : 'trend-bearish'}`;
            }

            // Update card metrics row
            const metricsRow = card.querySelector('.card-metrics-row');
            if (metricsRow) {
                metricsRow.innerHTML = `
                    <span>MAE: <b>${p.mae.toFixed(2)}$</b></span>
                    <span>MAPE: <b>${p.mape.toFixed(2)}%</b></span>
                    <span>R²: <b>${p.r2.toFixed(4)}</b></span>
                `;
            }
        });

        // Render 20-day Trajectory Fan Chart
        renderTrajectoryChart();
    } catch (err) {
        console.error("Lỗi tải dự báo đa chu kỳ:", err);
    }
}

// 3. Render Trajectory Fan Chart (20 Days Forecast Curve with Confidence Band)
function renderTrajectoryChart() {
    if (!multiHorizonCache || !multiHorizonCache.trajectory) return;

    const trajectory = multiHorizonCache.trajectory;
    const dates = trajectory.map(t => t.date);
    const predPrices = trajectory.map(t => t.prices[currentTrajProduct]);
    const lowerBounds = trajectory.map(t => t.lower_bounds[currentTrajProduct]);
    const upperBounds = trajectory.map(t => t.upper_bounds[currentTrajProduct]);

    const targetIndex = Math.min(currentHorizon - 1, trajectory.length - 1);
    const targetDate = dates[targetIndex];
    const targetPrice = predPrices[targetIndex];

    const prodNames = {
        'MG95': 'MOGAS 95 Unleaded',
        'MG92': 'MOGAS 92 Unleaded',
        'DO_0001': 'Gasoil 10ppm (DO 0.001%)',
        'DO_005': 'Gasoil 500ppm (DO 0.05%)'
    };

    const options = {
        series: [
            {
                name: `Dự Báo Quỹ Đạo AI (${currentTrajProduct})`,
                type: 'line',
                data: predPrices
            },
            {
                name: 'Biên Trên 95% (Upper Fan)',
                type: 'line',
                data: upperBounds
            },
            {
                name: 'Biên Dưới 95% (Lower Fan)',
                type: 'line',
                data: lowerBounds
            }
        ],
        chart: {
            type: 'line',
            height: 380,
            background: 'transparent',
            toolbar: { show: true },
            animations: { enabled: true, easing: 'easeinout', speed: 600 }
        },
        colors: ['#38bdf8', '#f59e0b', '#10b981'],
        stroke: {
            width: [3.5, 1.5, 1.5],
            curve: 'smooth',
            dashArray: [0, 4, 4]
        },
        fill: {
            type: ['solid', 'solid', 'solid']
        },
        xaxis: {
            categories: dates,
            labels: {
                style: { colors: '#94a3b8', fontSize: '11px', fontFamily: 'Inter' },
                rotate: -35
            },
            axisBorder: { color: 'rgba(255,255,255,0.1)' }
        },
        yaxis: {
            labels: {
                style: { colors: '#94a3b8', fontSize: '11px', fontFamily: 'JetBrains Mono' },
                formatter: val => `${val ? val.toFixed(1) : ''}$`
            }
        },
        tooltip: {
            theme: 'dark',
            x: { show: true },
            y: {
                formatter: val => `${val ? val.toFixed(2) : ''} USD/bbl`
            }
        },
        legend: {
            position: 'top',
            labels: { colors: '#cbd5e1' },
            fontFamily: 'Inter'
        },
        grid: {
            borderColor: 'rgba(255, 255, 255, 0.06)',
            strokeDashArray: 3
        },
        annotations: {
            points: [
                {
                    x: targetDate,
                    y: targetPrice,
                    marker: {
                        size: 7,
                        fillColor: '#ef4444',
                        strokeColor: '#fff',
                        strokeWidth: 2,
                        shape: 'circle'
                    },
                    label: {
                        borderColor: '#ef4444',
                        offsetY: -15,
                        style: {
                            color: '#fff',
                            background: '#ef4444',
                            fontWeight: 700,
                            fontSize: '11px'
                        },
                        text: `Mốc T+${currentHorizon}: ${targetPrice.toFixed(2)}$`
                    }
                }
            ]
        }
    };

    const container = document.getElementById('trajectoryChart');
    if (!container) return;

    if (trajectoryChart) {
        trajectoryChart.updateOptions(options);
    } else {
        trajectoryChart = new ApexCharts(container, options);
        trajectoryChart.render();
    }
}

// 4. Draw Mini Sparklines
function drawSparkline(canvas, points, color) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width = canvas.parentElement.clientWidth;
    const height = canvas.height = canvas.parentElement.clientHeight;

    ctx.clearRect(0, 0, width, height);

    if (!points || points.length < 2) return;

    const min = Math.min(...points);
    const max = Math.max(...points);
    const range = (max - min) || 1;

    const padding = 6;
    const h = height - padding * 2;
    const w = width - padding * 2;

    const getX = (i) => padding + (i / (points.length - 1)) * w;
    const getY = (v) => padding + h - ((v - min) / range) * h;

    const gradient = ctx.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, color === '#10b981' ? 'rgba(16, 185, 129, 0.25)' : 'rgba(244, 63, 94, 0.25)');
    gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');

    ctx.beginPath();
    ctx.moveTo(getX(0), getY(points[0]));
    for (let i = 1; i < points.length; i++) {
        ctx.lineTo(getX(i), getY(points[i]));
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = 2.2;
    ctx.lineCap = 'round';
    ctx.stroke();

    ctx.lineTo(getX(points.length - 1), height);
    ctx.lineTo(getX(0), height);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();
}

// 5. Load Main Historical & Prediction Chart
async function loadHistoricalChart() {
    try {
        const res = await fetch('/api/historical?limit=180');
        historicalDataCache = await res.json();
        renderMainChart();
    } catch (err) {
        console.error("Lỗi tải biểu đồ lịch sử:", err);
    }
}

function renderMainChart() {
    const dates = historicalDataCache.map(d => d.date);
    const actual = historicalDataCache.map(d => d[`actual_${currentProductFilter}`]);
    const pred = historicalDataCache.map(d => d[`pred_${currentProductFilter}`]);

    const options = {
        series: [
            { name: `Giá Thực tế (${currentProductFilter})`, data: actual },
            { name: `Dự báo AI (${currentProductFilter})`, data: pred }
        ],
        chart: {
            type: 'line',
            height: 380,
            background: 'transparent',
            toolbar: { show: true },
            animations: { enabled: true, easing: 'easeinout', speed: 800 }
        },
        colors: ['#0284c7', '#10b981'],
        stroke: { width: [2.5, 2.5], curve: 'smooth' },
        xaxis: {
            categories: dates,
            labels: {
                style: { colors: '#94a3b8', fontSize: '11px', fontFamily: 'Inter' },
                rotate: -45,
                rotateAlways: false
            },
            axisBorder: { color: 'rgba(255,255,255,0.1)' }
        },
        yaxis: {
            labels: {
                style: { colors: '#94a3b8', fontSize: '11px', fontFamily: 'JetBrains Mono' },
                formatter: val => `${val ? val.toFixed(1) : ''}$`
            }
        },
        tooltip: {
            theme: 'dark',
            x: { show: true },
            y: { formatter: val => `${val ? val.toFixed(2) : ''} USD/bbl` }
        },
        legend: {
            position: 'top',
            labels: { colors: '#cbd5e1' },
            fontFamily: 'Inter'
        },
        grid: {
            borderColor: 'rgba(255, 255, 255, 0.06)',
            strokeDashArray: 3
        }
    };

    const container = document.getElementById('mainChart');
    if (!container) return;

    if (mainChart) {
        mainChart.destroy();
    }
    mainChart = new ApexCharts(container, options);
    mainChart.render();
}

function updateMainChartSeries() {
    if (!mainChart || !historicalDataCache.length) return;
    const actual = historicalDataCache.map(d => d[`actual_${currentProductFilter}`]);
    const pred = historicalDataCache.map(d => d[`pred_${currentProductFilter}`]);

    mainChart.updateSeries([
        { name: `Giá Thực tế (${currentProductFilter})`, data: actual },
        { name: `Dự báo AI (${currentProductFilter})`, data: pred }
    ]);
}

// 6. Load Crack Spreads Chart
async function loadCrackSpreadChart() {
    try {
        const res = await fetch('/api/crack-spreads?limit=140');
        crackSpreadCache = await res.json();
        renderSpreadChart();
    } catch (err) {
        console.error("Lỗi tải Crack Spreads:", err);
    }
}

function renderSpreadChart() {
    const dates = crackSpreadCache.map(d => d.date);
    const spPremium = crackSpreadCache.map(d => d.spread_MG95_MG92);
    const spQuality = crackSpreadCache.map(d => d.spread_DO0001_DO005);
    const spCrack = crackSpreadCache.map(d => d.spread_GAS_OIL);

    const options = {
        series: [
            { name: 'Xăng Cao Cấp vs Tiêu Chuẩn (MG95 - MG92)', data: spPremium },
            { name: 'Chất Lượng Diesel (DO 0.001% - DO 0.05%)', data: spQuality },
            { name: 'Biên Lợi Nhuận Lọc Dầu (MG95 - DO 0.05%)', data: spCrack }
        ],
        chart: {
            type: 'area',
            height: 320,
            background: 'transparent',
            toolbar: { show: false }
        },
        colors: ['#38bdf8', '#fbbf24', '#f43f5e'],
        fill: {
            type: 'gradient',
            gradient: {
                shadeIntensity: 1,
                opacityFrom: 0.35,
                opacityTo: 0.05,
                stops: [0, 95, 100]
            }
        },
        stroke: { width: 2, curve: 'smooth' },
        xaxis: {
            categories: dates,
            labels: {
                style: { colors: '#94a3b8', fontSize: '10px' },
                rotate: -45
            }
        },
        yaxis: {
            labels: {
                style: { colors: '#94a3b8', fontSize: '10px' },
                formatter: val => `${val ? val.toFixed(1) : ''}$`
            }
        },
        tooltip: {
            theme: 'dark',
            y: { formatter: val => `${val ? val.toFixed(2) : ''} USD/bbl` }
        },
        legend: {
            position: 'top',
            labels: { colors: '#cbd5e1' }
        },
        grid: {
            borderColor: 'rgba(255, 255, 255, 0.06)',
            strokeDashArray: 3
        }
    };

    const container = document.getElementById('spreadChart');
    if (!container) return;

    if (spreadChart) spreadChart.destroy();
    spreadChart = new ApexCharts(container, options);
    spreadChart.render();
}

// 7. Load Benchmark Metrics Table
async function loadMetricsTable() {
    try {
        const res = await fetch('/api/metrics');
        const data = await res.json();
        const tbody = document.getElementById('metricsTableBody');
        if (!tbody) return;

        tbody.innerHTML = '';

        // Render models comparison
        if (data.models_comparison) {
            data.models_comparison.forEach(m => {
                const tr = document.createElement('tr');
                const isBest = m.Model.includes('Ensemble') || m.Model.includes('Residual');
                if (isBest) tr.className = 'highlight-row';

                tr.innerHTML = `
                    <td><b>${m.Model}</b> ${isBest ? '<span class="trend-badge trend-bullish" style="font-size: 0.65rem; margin-left: 6px;">SOTA</span>' : ''}</td>
                    <td>${m.Avg_MAE.toFixed(2)}$</td>
                    <td>${m.Avg_RMSE.toFixed(2)}$</td>
                    <td>${m.Avg_MAPE.toFixed(2)}%</td>
                    <td><b>${m.Avg_R2.toFixed(4)}</b></td>
                `;
                tbody.appendChild(tr);
            });
        }
    } catch (err) {
        console.error("Lỗi tải bảng metrics:", err);
    }
}

// 8. Load Vietnam Retail Fuel Forecast
async function loadVietnamForecast(fxRate = 25400) {
    try {
        const res = await fetch(`/api/vietnam/forecast?horizon=${currentHorizon}&fx_rate=${fxRate}`);
        const data = await res.json();
        vietnamForecastCache = data;

        // Update banner text
        const bannerSub = document.querySelector('.vn-cycle-title p');
        if (bannerSub) {
            bannerSub.textContent = `🎯 ${data.executive_summary} (Kỳ Thứ Năm: ${data.next_adjustment_date} · Tỷ giá: ${data.usd_vnd_rate.toLocaleString('vi-VN')} VND/USD)`;
        }

        // Update each product card in Vietnam grid
        data.products.forEach(p => {
            const card = document.getElementById(`card-vn-${p.product_code}`);
            if (!card) return;

            // Current price
            const priceMain = card.querySelector('.vn-price-main');
            if (priceMain) {
                priceMain.textContent = `${p.current_retail_vnd.toLocaleString('vi-VN')} đ/lít`;
            }

            // Forecast Label & Value
            const forecastLabel = card.querySelector('.forecast-label');
            if (forecastLabel) forecastLabel.textContent = `Dự báo T+${data.horizon}:`;

            const forecastVal = card.querySelector('.forecast-val');
            if (forecastVal) {
                forecastVal.textContent = `${p.predicted_retail_vnd.toLocaleString('vi-VN')} đ`;
            }

            // Delta pill
            const deltaPill = card.querySelector('.delta-pill');
            if (deltaPill) {
                const sign = p.delta_vnd >= 0 ? '+' : '';
                deltaPill.textContent = `${sign}${p.delta_vnd.toLocaleString('vi-VN')} đ`;
                deltaPill.className = `delta-pill ${p.delta_vnd >= 0 ? 'delta-pos' : 'delta-neg'}`;
            }

            // Cycle alert
            const alertBadge = card.querySelector('.vn-cycle-alert');
            if (alertBadge) {
                alertBadge.textContent = p.cycle_signal;
                if (p.action_class === 'bullish') {
                    alertBadge.className = 'vn-cycle-alert vn-alert-increase';
                } else if (p.action_class === 'bearish') {
                    alertBadge.className = 'vn-cycle-alert vn-alert-decrease';
                } else {
                    alertBadge.className = 'vn-cycle-alert vn-alert-neutral';
                }
            }

            // Draw sparkline canvas in VND
            const canvas = card.querySelector('.sparkline-canvas');
            if (canvas && p.sparkline_vnd) {
                drawSparkline(canvas, p.sparkline_vnd, p.delta_vnd >= 0 ? '#10b981' : '#f43f5e');
            }

            // Attach tax button click
            const btnTax = card.querySelector('.btn-show-tax');
            if (btnTax) {
                btnTax.onclick = () => showTaxBreakdownModal(p.product_code);
            }
        });
    } catch (err) {
        console.error("Lỗi tải dự báo giá xăng dầu Việt Nam:", err);
    }
}

// 9. Show Tax Breakdown Modal
function showTaxBreakdownModal(productCode) {
    if (!vietnamForecastCache) return;
    const p = vietnamForecastCache.products.find(item => item.product_code === productCode);
    if (!p) return;

    const modal = document.getElementById('taxModal');
    const title = document.getElementById('taxModalTitle');
    const body = document.getElementById('taxModalBody');

    if (!modal || !title || !body) return;

    title.textContent = `Cơ Cấu Giá Bán Lẻ: ${p.vietnam_name}`;

    const c = p.components;
    const total = c.retail_vnd;
    const pct = (val) => ((val / total) * 100).toFixed(1);

    body.innerHTML = `
        <div style="margin-bottom: 14px; font-size: 0.82rem; color: #94a3b8;">
            Giá CIF thị trường Singapore (Mốc T+${currentHorizon}): <b style="color: #38bdf8;">${p.singapore_predicted_usd_bbl.toFixed(2)} USD/thùng</b> 
            (Quy đổi theo tỷ giá: <b>${vietnamForecastCache.usd_vnd_rate.toLocaleString('vi-VN')} VND/USD</b>)
        </div>
        <div class="tax-row">
            <span>1. Giá CIF nhập khẩu quy đổi (đã gồm cước biển & bảo hiểm):</span>
            <span><b>${c.cif_vnd.toLocaleString('vi-VN')} đ</b> (${pct(c.cif_vnd)}%)</span>
        </div>
        <div class="tax-row">
            <span>2. Thuế Nhập khẩu (Import Duty):</span>
            <span><b>${c.import_duty_vnd.toLocaleString('vi-VN')} đ</b> (${pct(c.import_duty_vnd)}%)</span>
        </div>
        <div class="tax-row">
            <span>3. Thuế Tiêu thụ đặc biệt (Excise Tax):</span>
            <span><b>${c.excise_tax_vnd.toLocaleString('vi-VN')} đ</b> (${pct(c.excise_tax_vnd)}%)</span>
        </div>
        <div class="tax-row">
            <span>4. Thuế Bảo vệ môi trường (Nghị quyết 42/2023):</span>
            <span><b>${c.env_tax_vnd.toLocaleString('vi-VN')} đ</b> (${pct(c.env_tax_vnd)}%)</span>
        </div>
        <div class="tax-row">
            <span>5. Chi phí kinh doanh định mức &amp; Lợi nhuận định mức:</span>
            <span><b>${c.operating_cost_vnd.toLocaleString('vi-VN')} đ</b> (${pct(c.operating_cost_vnd)}%)</span>
        </div>
        <div class="tax-row">
            <span>6. Thuế Giá trị gia tăng VAT (10%):</span>
            <span><b>${c.vat_vnd.toLocaleString('vi-VN')} đ</b> (${pct(c.vat_vnd)}%)</span>
        </div>
        <div class="tax-row">
            <span>👉 TỔNG GIÁ BÁN LẺ DỰ KIẾN (PETROLIMEX):</span>
            <span style="font-size: 1.15rem; color: #fef08a;">${total.toLocaleString('vi-VN')} đ/lít</span>
        </div>
    `;

    modal.classList.add('active');
}

// 10. Live Forecast Modal (Deep Dive)
async function showLiveForecastModal() {
    try {
        const res = await fetch(`/api/forecast/multi-horizon?horizon=${currentHorizon}`);
        const data = await res.json();

        const modal = document.getElementById('forecastModal');
        const dateEl = document.getElementById('modalTargetDate');
        const grid = document.getElementById('modalForecastGrid');

        if (dateEl) dateEl.textContent = `${data.target_date} (Chân trời T+${data.horizon})`;
        if (!grid) return;

        grid.innerHTML = '';

        data.predictions.forEach(p => {
            const card = document.createElement('div');
            card.className = 'modal-pred-card';
            const isPos = p.delta >= 0;

            card.innerHTML = `
                <div class="title">${p.label}</div>
                <div class="val">${p.predicted_price.toFixed(2)}$</div>
                <div class="delta ${isPos ? 'delta-pos' : 'delta-neg'}">
                    ${isPos ? '+' : ''}${p.delta.toFixed(2)}$ (${isPos ? '+' : ''}${p.delta_pct.toFixed(2)}%)
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 6px;">
                    Khoảng tin cậy 95%: <b>${p.confidence_lower.toFixed(2)}$</b> – <b>${p.confidence_upper.toFixed(2)}$</b>
                </div>
                <div style="font-size: 0.75rem; margin-top: 4px; color: ${isPos ? '#34d399' : '#f87171'}; font-weight: 700;">
                    Tín hiệu: ${p.signal}
                </div>
            `;
            grid.appendChild(card);
        });

        modal.classList.add('active');
    } catch (err) {
        console.error("Lỗi mở modal dự báo:", err);
    }
}

// 11. Run Stress Simulation
async function runStressSimulation() {
    const gasShock = parseFloat(document.getElementById('gasShockSlider').value) || 0;
    const dieselShock = parseFloat(document.getElementById('dieselShockSlider').value) || 0;
    const vol = parseFloat(document.getElementById('volSlider').value) || 1.0;

    try {
        const res = await fetch('/api/forecast/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                shock_gasoline_pct: gasShock,
                shock_diesel_pct: dieselShock,
                volatility_multiplier: vol
            })
        });

        const data = await res.json();
        const container = document.getElementById('simResultCards');
        if (!container) return;

        container.innerHTML = '';
        data.predictions.forEach(p => {
            const card = document.createElement('div');
            card.className = 'sim-result-card';
            const isPos = p.delta >= 0;

            card.innerHTML = `
                <div class="name">${p.label}</div>
                <div class="price">${p.predicted_price.toFixed(2)}$</div>
                <div class="diff ${isPos ? 'diff-pos' : 'diff-neg'}">
                    ${isPos ? '+' : ''}${p.delta.toFixed(2)}$ (${isPos ? '+' : ''}${p.delta_pct.toFixed(2)}%)
                </div>
            `;
            container.appendChild(card);
        });
    } catch (err) {
        console.error("Lỗi chạy kịch bản giả lập:", err);
    }
}
