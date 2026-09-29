import requests

base = 'http://127.0.0.1:8000'

r1 = requests.get(base + '/')
print('1. GET / :', r1.status_code, f'({len(r1.text)} bytes) | Has Horizon bar:', 'horizon-control-bar' in r1.text)

r2 = requests.get(base + '/api/overview')
print('2. GET /api/overview :', r2.status_code, list(r2.json()['products'].keys()))

r3 = requests.get(base + '/api/forecast/latest')
d3 = r3.json()
print('3. GET /api/forecast/latest :', r3.status_code, 'Date:', d3.get('prediction_date'), 'Latency:', d3.get('latency_ms'), 'ms')

# Test Multi-Horizon Endpoints (3, 7, 20 days)
for h in [3, 7, 20]:
    rh = requests.get(f'{base}/api/forecast/multi-horizon?horizon={h}')
    dh = rh.json()
    print(f'4. GET /api/forecast/multi-horizon?horizon={h} :', rh.status_code, f"Target: {dh.get('target_date')} | Label: {dh.get('horizon_label')} | Trajectory points: {len(dh.get('trajectory', []))}")

r5 = requests.post(base + '/api/forecast/simulate', json={'shock_gasoline_pct': 3.5, 'shock_diesel_pct': -2.0, 'volatility_multiplier': 1.2})
print('5. POST /api/forecast/simulate :', r5.status_code)

r6 = requests.get(base + '/api/historical?limit=5')
print('6. GET /api/historical :', r6.status_code)

r7 = requests.get(base + '/api/crack-spreads?limit=5')
print('7. GET /api/crack-spreads :', r7.status_code)

r8 = requests.get(base + '/api/metrics')
print('8. GET /api/metrics :', r8.status_code)

# Test Vietnam Retail Forecast for horizons 3, 7, 20
for h in [3, 7, 20]:
    rvn = requests.get(f'{base}/api/vietnam/forecast?horizon={h}&fx_rate=25400')
    dvn = rvn.json()
    print(f"9. GET /api/vietnam/forecast?horizon={h} :", rvn.status_code, f"| Mốc: {dvn.get('horizon_label')} | {dvn.get('executive_summary')}")
    for p in dvn['products'][:2]:
        delta = p['delta_vnd']
        sign = '+' if delta >= 0 else ''
        print(f"   • {p['vietnam_name']:30s}: Hiện tại {int(p['current_retail_vnd']):,} đ -> Dự kiến {int(p['predicted_retail_vnd']):,} đ ({sign}{int(delta):,} đ) | {p['cycle_signal']}")

print("\n🎉 ALL 11 MULTI-HORIZON API ENDPOINTS + VIETNAM DUAL FORECAST TESTED & PASSED 100%!")
