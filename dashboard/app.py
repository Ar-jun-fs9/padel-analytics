"""
Analytics Dashboard (Flask)
Displays interactive charts of shot classifications and game analytics.
Run with: python main.py --dashboard
"""

import json
import os
from pathlib import Path
from flask import Flask, render_template_string, jsonify


app = Flask(__name__)
OUTPUT_DIR = Path("output")

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Padel Analytics Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg: #0d0f14;
      --card: #161923;
      --border: #252a38;
      --accent: #e84040;
      --text: #e2e8f0;
      --muted: #64748b;
      --green: #22c55e;
      --blue: #3b82f6;
      --yellow: #eab308;
      --purple: #a855f7;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Segoe UI', system-ui, sans-serif;
      min-height: 100vh;
    }
    header {
      background: var(--card);
      border-bottom: 1px solid var(--border);
      padding: 1.2rem 2rem;
      display: flex;
      align-items: center;
      gap: 1rem;
    }
    header h1 { font-size: 1.5rem; font-weight: 700; }
    header span { color: var(--accent); }
    .badge {
      background: var(--accent);
      color: white;
      font-size: 0.7rem;
      padding: 2px 10px;
      border-radius: 99px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    main {
      max-width: 1200px;
      margin: 0 auto;
      padding: 2rem;
    }
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
    }
    .kpi {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.2rem 1.4rem;
    }
    .kpi-label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
    .kpi-value { font-size: 2rem; font-weight: 800; margin-top: 4px; }
    .kpi-value.green { color: var(--green); }
    .kpi-value.blue  { color: var(--blue); }
    .kpi-value.yellow{ color: var(--yellow);}
    .kpi-value.red   { color: var(--accent);}
    .charts-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.5rem;
      margin-bottom: 2rem;
    }
    @media (max-width: 700px) { .charts-grid { grid-template-columns: 1fr; } }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.5rem;
    }
    .card h2 { font-size: 0.9rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 1rem; }
    .shot-list { list-style: none; }
    .shot-list li {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0.6rem 0;
      border-bottom: 1px solid var(--border);
      font-size: 0.9rem;
    }
    .shot-list li:last-child { border: none; }
    .pill {
      background: #1e2535;
      border-radius: 6px;
      padding: 2px 10px;
      font-size: 0.75rem;
      font-weight: 600;
    }
    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
    th { color: var(--muted); text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.06em; padding: 0.5rem 0.8rem; text-align: left; border-bottom: 1px solid var(--border); }
    td { padding: 0.5rem 0.8rem; border-bottom: 1px solid #1a1f2e; }
    tr:hover td { background: #1e2535; }
    .tag {
      display: inline-block;
      border-radius: 4px;
      padding: 1px 8px;
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
    }
    .FOREHAND     { background: #14532d; color: #4ade80; }
    .BACKHAND     { background: #1e3a5f; color: #60a5fa; }
    .SMASH        { background: #4c1d1d; color: #f87171; }
    .VOLLEY       { background: #3f2a00; color: #fbbf24; }
    .LOFT-BANDEJA { background: #3b1f5e; color: #c084fc; }
    footer {
      text-align: center;
      color: var(--muted);
      font-size: 0.75rem;
      padding: 2rem;
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Padel <span>Analytics</span></h1>
      <div style="font-size:0.8rem;color:var(--muted);margin-top:2px;">Shot Classification System</div>
    </div>
    <span class="badge">Layman AI</span>
  </header>
  <main>
    <div class="kpi-grid" id="kpis"></div>
    <div class="charts-grid">
      <div class="card">
        <h2>Shot Distribution</h2>
        <canvas id="donutChart" height="220"></canvas>
      </div>
      <div class="card">
        <h2>Shot Timeline</h2>
        <canvas id="timelineChart" height="220"></canvas>
      </div>
    </div>
    <div class="card" style="margin-bottom:1.5rem">
      <h2>Shot Breakdown</h2>
      <ul class="shot-list" id="shotList"></ul>
    </div>
    <div class="card">
      <h2>All Detected Shots</h2>
      <div class="table-wrap">
        <table id="shotsTable">
          <thead>
            <tr>
              <th>#</th><th>Frame</th><th>Time (s)</th>
              <th>Shot Type</th><th>Player</th>
              <th>Speed</th><th>Direction</th><th>Confidence</th>
            </tr>
          </thead>
          <tbody id="shotsBody"></tbody>
        </table>
      </div>
    </div>
  </main>
  <footer>Padel Game Analytics · Layman AI Internship Assignment</footer>

  <script>
    const PALETTE = {
      FOREHAND:'#22c55e', BACKHAND:'#3b82f6', SMASH:'#ef4444',
      VOLLEY:'#eab308', 'LOFT/BANDEJA':'#a855f7'
    };

    async function load() {
      const r = await fetch('/api/data');
      const d = await r.json();
      const summary = d.summary || {};
      const shots   = d.shots || [];
      const counts  = summary.shot_counts || {};
      const total   = summary.total_shots || 0;

      // KPIs
      const kpiData = [
        { label:'Total Shots',    value: total,                    cls:'green'  },
        { label:'Forehands',      value: counts.FOREHAND||0,       cls:'green'  },
        { label:'Backhands',      value: counts.BACKHAND||0,       cls:'blue'   },
        { label:'Smashes',        value: counts.SMASH||0,          cls:'red'    },
        { label:'Volleys',        value: counts.VOLLEY||0,         cls:'yellow' },
        { label:'Bounces',        value: summary.total_bounces||0, cls:'yellow' },
      ];
      document.getElementById('kpis').innerHTML = kpiData.map(k =>
        `<div class="kpi"><div class="kpi-label">${k.label}</div>
         <div class="kpi-value ${k.cls}">${k.value}</div></div>`).join('');

      // Donut
      const labels = Object.keys(counts);
      const values = Object.values(counts);
      new Chart(document.getElementById('donutChart'), {
        type: 'doughnut',
        data: {
          labels,
          datasets: [{ data: values, backgroundColor: labels.map(l => PALETTE[l]||'#64748b'),
            borderColor:'#0d0f14', borderWidth:3 }]
        },
        options: { plugins:{ legend:{ labels:{ color:'#e2e8f0', font:{size:12} } } }, cutout:'60%' }
      });

      // Timeline
      const shotTypes = [...new Set(shots.map(s => s.shot_type))];
      const datasets = shotTypes.map(st => ({
        label: st,
        data: shots.filter(s => s.shot_type === st).map(s => ({ x: s.timestamp, y: s.ball_speed||1 })),
        backgroundColor: PALETTE[st]||'#64748b',
        pointRadius: 6,
      }));
      new Chart(document.getElementById('timelineChart'), {
        type: 'scatter',
        data: { datasets },
        options: {
          scales: {
            x:{ title:{ display:true, text:'Time (s)', color:'#64748b' }, ticks:{color:'#64748b'}, grid:{color:'#1a1f2e'} },
            y:{ title:{ display:true, text:'Ball Speed', color:'#64748b' }, ticks:{color:'#64748b'}, grid:{color:'#1a1f2e'} }
          },
          plugins:{ legend:{ labels:{ color:'#e2e8f0', font:{size:11} } } }
        }
      });

      // Shot list
      const pct = summary.shot_percentages || {};
      document.getElementById('shotList').innerHTML = labels.map(l =>
        `<li><span>${l}</span>
          <span style="display:flex;gap:0.8rem;align-items:center">
            <span class="pill" style="color:${PALETTE[l]||'#fff'}">${counts[l]}</span>
            <span style="color:var(--muted);font-size:0.8rem">${pct[l]||0}%</span>
          </span></li>`).join('');

      // Table
      const tbody = document.getElementById('shotsBody');
      tbody.innerHTML = shots.map((s,i) => {
        const cls = (s.shot_type||'').replace('/','_').replace(' ','-');
        return `<tr>
          <td>${i+1}</td>
          <td>${s.frame}</td>
          <td>${(s.timestamp||0).toFixed(2)}</td>
          <td><span class="tag ${s.shot_type}">${s.shot_type||'—'}</span></td>
          <td>${s.player_id != null ? 'P'+s.player_id : '—'}</td>
          <td>${(s.ball_speed||0).toFixed(1)} px/f</td>
          <td>${s.direction||'—'}</td>
          <td>${((s.confidence||0)*100).toFixed(0)}%</td>
        </tr>`;
      }).join('');
    }
    load();
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/data")
def api_data():
    json_path = OUTPUT_DIR / "shots.json"
    if json_path.exists():
        with open(json_path) as f:
            return jsonify(json.load(f))
    return jsonify({"summary": {}, "shots": [], "metadata": {}})


def run_dashboard(output_dir="output"):
    global OUTPUT_DIR
    OUTPUT_DIR = Path(output_dir)
    print("\n[INFO] Starting Analytics Dashboard …")
    print("   Open: http://127.0.0.1:5050")
    print("   Press Ctrl+C to stop\n")
    app.run(host="0.0.0.0", port=5050, debug=False)
