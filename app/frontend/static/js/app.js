// app/frontend/static/js/app.js

document.addEventListener('DOMContentLoaded', () => {
    initDashboard();
});

let mainChart = null;
let spreadChart = null;
let currentProductFilter = 'MG95';
let historicalDataCache = [];
let crackSpreadCache = [];
let vietnamForecastCache = null;

async function initDashboard() {
    setupEventListeners();
    await Promise.all([
        loadMarketOverview(),
        loadHistoricalChart(),
        loadCrackSpreadChart(),
        loadMetricsTable(),
        loadVietnamForecast()
    ]);
}

function setupEventListeners() {
    // Live forecast modal button
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

    // Market Mode Switcher (Singapore vs Vietnam)
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

    // Tax Modal close
    const btnCloseTaxModal = document.getElementById('btnCloseTaxModal');
    const taxModal = document.getElementById('taxModal');
    if (btnCloseTaxModal && taxModal) {
        btnCloseTaxModal.addEventListener('click', () => taxModal.classList.remove('active'));
        taxModal.addEventListener('click', (e) => {
            if (e.target === taxModal) taxModal.classList.remove('active');
        });
    }

    // Update Vietnam FX
    const btnUpdateVnFx = document.getElementById('btnUpdateVnFx');
    if (btnUpdateVnFx) {
        btnUpdateVnFx.addEventListener('click', () => {
            const fx = parseFloat(document.getElementById('inputVnFx').value) || 25400;
            loadVietnamForecast(fx);
        });
    }

    // Chart product filter buttons
    document.querySelectorAll('.btn-filter[data-product]').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.btn-filter[data-product]').forEach(b => b.classList.remove('active'));
            e.currentTarget.classList.add('active');
            currentProductFilter = e.currentTarget.getAttribute('data-product');
            updateMainChartSeries();
        });
    });

    // Simulation Sliders
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

// 1. Load Market Overview & KPI Cards
async function loadMarketOverview() {
    try {
        const res = await fetch('/api/overview');
        const data = await res.json();

        // Update header info
        const dateEl = document.getElementById('headerLastDate');
        if (dateEl) dateEl.textContent = data.last_updated;

        const hwEl = document.getElementById('hardwareTag');
        if (hwEl) hwEl.textContent = data.hardware;

        // Render each product card
        const products = ['MG95', 'MG92', 'DO_0001', 'DO_005'];
        products.forEach(p => {
            const pData = data.products[p];
            if (!pData) return;

            const card = document.getElementById(`card-${p}`);
            if (!card) return;

            card.querySelector('.price-main').textContent = `${pData.current_price.toFixed(2)}$`;
            
            const forecastVal = card.querySelector('.forecast-val');
            if (forecastVal) forecastVal.textContent = `${pData.predicted_next_price.toFixed(2)}$`;

            const deltaPill = card.querySelector('.delta-pill');
            if (deltaPill) {
                const sign = pData.delta_predicted >= 0 ? '+' : '';
                deltaPill.textContent = `${sign}${pData.delta_predicted.toFixed(2)}$ (${sign}${pData.delta_pct.toFixed(2)}%)`;
                deltaPill.className = `delta-pill ${pData.delta_predicted >= 0 ? 'delta-pos' : 'delta-neg'}`;
            }

            const trendBadge = card.querySelector('.trend-badge');
            if (trendBadge) {
                trendBadge.textContent = pData.trend === 'bullish' ? '▲ TĂNG' : (pData.trend === 'bearish' ? '▼ GIẢM' : '◆ ỔN ĐỊNH');
                trendBadge.className = `trend-badge ${pData.trend === 'bullish' ? 'trend-bullish' : 'trend-bearish'}`;
            }

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

// 2. Draw Mini Sparklines
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

    // Gradient fill under line
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

    // Fill
    ctx.lineTo(getX(points.length - 1), height);
    ctx.lineTo(getX(0), height);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();
}

// 3. Load Main Historical & Prediction Chart
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
            toolbar: { show: true, tools: { download: true, zoom: true, reset: true } },
            animations: { enabled: true, easing: 'easeinout', speed: 800 }
        },
        colors: ['#ffffff', '#38bdf8'],
        stroke: {
            curve: 'smooth',
            width: [2, 2.5],
            dashArray: [0, 4]
        },
        xaxis: {
            categories: dates,
            labels: { style: { colors: '#64748b', fontSize: '11px' }, rotate: -35 },
            axisBorder: { color: '#334155' },
            tickAmount: 12
        },
        yaxis: {
            labels: {
                formatter: (val) => `${val.toFixed(1)}$`,
                style: { colors: '#64748b', fontSize: '11px' }
            }
        },
        grid: {
            borderColor: 'rgba(255, 255, 255, 0.05)',
            strokeDashArray: 3
        },
        legend: {
            labels: { colors: '#cbd5e1' },
            position: 'top',
            horizontalAlign: 'right'
        },
        tooltip: {
            theme: 'dark',
            x: { show: true },
            y: {
                formatter: (val) => `${val.toFixed(2)} USD/thùng`
            }
        }
    };

    if (mainChart) {
        mainChart.destroy();
    }
    mainChart = new ApexCharts(document.getElementById('mainChart'), options);
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

// 4. Load Crack Spreads Chart
async function loadCrackSpreadChart() {
    try {
        const res = await fetch('/api/crack-spreads?limit=140');
        crackSpreadCache = await res.json();

        const dates = crackSpreadCache.map(d => d.date);
        const sPremium = crackSpreadCache.map(d => d.spread_MG95_MG92);
        const sQuality = crackSpreadCache.map(d => d.spread_DO0001_DO005);
        const sCrack = crackSpreadCache.map(d => d.spread_GAS_OIL);

        const options = {
            series: [
                { name: 'Premium Spread (MG95 - MG92)', data: sPremium },
                { name: 'Quality Spread (DO 0.001% - DO 0.05%)', data: sQuality },
                { name: 'Refining Margin (MG95 - DO 0.05%)', data: sCrack }
            ],
            chart: {
                type: 'area',
                height: 320,
                background: 'transparent',
                toolbar: { show: false }
            },
            colors: ['#f43f5e', '#38bdf8', '#10b981'],
            fill: {
                type: 'gradient',
                gradient: { shadeIntensity: 1, opacityFrom: 0.35, opacityTo: 0.02, stops: [0, 95] }
            },
            stroke: { curve: 'smooth', width: 2 },
            xaxis: {
                categories: dates,
                labels: { style: { colors: '#64748b', fontSize: '11px' }, rotate: -35 },
                tickAmount: 10
            },
            yaxis: {
                labels: {
                    formatter: (val) => `${val.toFixed(2)}$`,
                    style: { colors: '#64748b', fontSize: '11px' }
                }
            },
            grid: { borderColor: 'rgba(255, 255, 255, 0.05)' },
            legend: { labels: { colors: '#cbd5e1' }, position: 'top' },
            tooltip: { theme: 'dark', y: { formatter: (v) => `${v.toFixed(3)} $/bbl` } }
        };

        if (spreadChart) spreadChart.destroy();
        spreadChart = new ApexCharts(document.getElementById('spreadChart'), options);
        spreadChart.render();
    } catch (err) {
        console.error("Lỗi tải biểu đồ Crack Spreads:", err);
    }
}

// 5. Load Metrics Comparison Table
async function loadMetricsTable() {
    try {
        const res = await fetch('/api/metrics');
        const data = await res.json();

        const tbody = document.getElementById('metricsTableBody');
        if (!tbody) return;
        tbody.innerHTML = '';

        data.models_comparison.forEach(m => {
            const tr = document.createElement('tr');
            const isWinner = m.Model.includes('Residual_GRU') || m.Model.includes('Ensemble');
            tr.innerHTML = `
                <td><b>${m.Model}</b> ${isWinner ? '<span class="tag-highlight">Top 1</span>' : ''}</td>
                <td><b style="color: #34d399">${m.MAE.toFixed(4)}$</b></td>
                <td>${m.RMSE.toFixed(4)}$</td>
                <td><span style="color: #38bdf8">${m['MAPE%'].toFixed(2)}%</span></td>
                <td><b>${m.R2.toFixed(4)}</b></td>
            `;
            tbody.appendChild(tr);
        });
    } catch (err) {
        console.error("Lỗi tải Metrics Table:", err);
    }
}

// 6. Live Next-Day Forecast Modal
async function showLiveForecastModal() {
    const modal = document.getElementById('forecastModal');
    if (!modal) return;
    modal.classList.add('active');

    const container = document.getElementById('modalForecastList');
    container.innerHTML = '<div style="text-align: center; padding: 24px; color: #94a3b8;">Đang kích hoạt Inference GPU NVIDIA RTX 4060 Ti...</div>';

    try {
        const res = await fetch('/api/forecast/latest');
        const data = await res.json();

        document.getElementById('modalTargetDate').textContent = data.prediction_date;
        document.getElementById('modalLatency').textContent = `${data.latency_ms} ms`;

        let html = '';
        data.predictions.forEach(p => {
            const sign = p.delta >= 0 ? '+' : '';
            const deltaColor = p.delta >= 0 ? '#34d399' : '#fb7185';
            html += `
                <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 14px 18px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-weight: 700; color: #fff; font-size: 0.95rem;">${p.product} — <span style="font-size: 0.8rem; color: #94a3b8; font-weight: 400;">${p.label}</span></div>
                        <div style="font-size: 0.78rem; color: #64748b; margin-top: 4px;">Hiện tại: <b>${p.current_price.toFixed(2)}$</b> | Khoảng tin cậy 95%: [${p.confidence_lower.toFixed(2)}$ – ${p.confidence_upper.toFixed(2)}$]</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-family: var(--font-mono); font-size: 1.25rem; font-weight: 800; color: #38bdf8;">${p.predicted_price.toFixed(2)}$</div>
                        <div style="font-size: 0.8rem; font-weight: 700; color: ${deltaColor};">${sign}${p.delta.toFixed(2)}$ (${sign}${p.delta_pct.toFixed(2)}%)</div>
                        <div style="font-size: 0.72rem; padding: 2px 8px; border-radius: 4px; background: rgba(56, 189, 248, 0.15); color: #38bdf8; display: inline-block; margin-top: 4px; font-weight: 600;">${p.signal}</div>
                    </div>
                </div>
            `;
        });
        container.innerHTML = html;
    } catch (err) {
        container.innerHTML = `<div style="color: #fb7185; text-align: center;">Lỗi gọi API: ${err.message}</div>`;
    }
}

// 7. Stress Testing Simulation
async function runStressSimulation() {
    const gasShock = parseFloat(document.getElementById('gasShockSlider').value);
    const dieselShock = parseFloat(document.getElementById('dieselShockSlider').value);
    const vol = parseFloat(document.getElementById('volSlider').value);

    const resBox = document.getElementById('simResultsContainer');
    resBox.innerHTML = '<div style="font-size: 0.8rem; color: #94a3b8; text-align: center; padding: 10px;">Đang tính toán ma trận sốc...</div>';

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

        let html = '';
        data.predictions.forEach(p => {
            const sign = p.delta >= 0 ? '+' : '';
            const color = p.delta >= 0 ? '#34d399' : '#fb7185';
            html += `
                <div class="sim-item">
                    <span class="p-name">${p.product}</span>
                    <span style="color: #94a3b8; font-size: 0.75rem;">Hiện: ${p.current_price.toFixed(2)}$</span>
                    <span class="p-pred">${p.predicted_price.toFixed(2)}$ (${sign}${p.delta_pct.toFixed(1)}%)</span>
                </div>
            `;
        });
        resBox.innerHTML = html;
    } catch (err) {
        resBox.innerHTML = `<div style="color: #fb7185; font-size: 0.75rem;">Lỗi mô phỏng: ${err.message}</div>`;
    }
}

// 8. Load Vietnam Petroleum Forecast
async function loadVietnamForecast(fxRate = 25400) {
    try {
        const res = await fetch(`/api/vietnam/forecast?fx_rate=${fxRate}`);
        const data = await res.json();
        vietnamForecastCache = data;

        // Update cycle date
        const dateEl = document.getElementById('vnNextCycleDate');
        if (dateEl) dateEl.textContent = data.next_adjustment_date;

        data.products.forEach(p => {
            const card = document.getElementById(`card-vn-${p.product_code}`);
            if (!card) return;

            // Current price
            const priceMain = card.querySelector('.vn-price-main');
            if (priceMain) {
                priceMain.textContent = `${p.current_retail_vnd.toLocaleString('vi-VN')} đ/lít`;
            }

            // Forecast T+1
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
            Giá CIF thị trường Singapore: <b style="color: #38bdf8;">${p.singapore_predicted_usd_bbl.toFixed(2)} USD/thùng</b> 
            (Quy đổi theo tỷ giá: <b>${vietnamForecastCache.fx_rate.toLocaleString('vi-VN')} VND/USD</b>)
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
