!pip install plotly --quiet
 
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load Cleaned Data
df = pd.read_csv('diabetic_cleaned.csv')

print("Shape:", df.shape)
print(df.head(3))

PALETTE = {
    'bg':       '#FAFAF8',    # warm off-white background
    'panel':    '#FFFFFF',    # chart card white
    'navy':     '#1B2A4A',    # titles / primary text
    'coral_dk': '#8B1A0A',    # highest value bars
    'coral':    '#C0392B',    # accent / peak marker
    'coral_md': '#E07850',    # mid value
    'coral_lt': '#F5C4B0',    # lowest value
    'steel':    '#2E6DA4',    # NOT readmitted (blue)
    'amber':    '#D4860B',    # >30d readmitted (amber)
    'grid':     '#EBEBEB',    # gridlines
    'sub':      '#6B7280',    # axis subtitles
}
 
FONT = 'Times New Roman'
 
def _hex_lerp(c1, c2, t):
    """Linearly interpolate between two hex colours."""
    r1,g1,b1 = int(c1[1:3],16), int(c1[3:5],16), int(c1[5:7],16)
    r2,g2,b2 = int(c2[1:3],16), int(c2[3:5],16), int(c2[5:7],16)
    r = int(r1 + (r2-r1)*t)
    g = int(g1 + (g2-g1)*t)
    b = int(b1 + (b2-b1)*t)
    return f'#{r:02X}{g:02X}{b:02X}'
 
def coral_gradient(values):
    """Map a list of floats to coral-scale hex colours."""
    mn, mx = min(values), max(values)
    stops = ['#F9D5C8','#F0A88A','#E07850','#C84A2A','#8B1A0A']
    colors = []
    for v in values:
        norm = (v - mn) / (mx - mn) if mx > mn else 0.5
        seg  = norm * (len(stops)-1)
        lo   = int(seg); hi = min(lo+1, len(stops)-1)
        colors.append(_hex_lerp(stops[lo], stops[hi], seg-lo))
    return colors
 
def base_layout(title, subtitle='', height=500):
    return dict(
        height=height,
        plot_bgcolor=PALETTE['panel'],
        paper_bgcolor=PALETTE['bg'],
        font=dict(family=FONT, color=PALETTE['navy'], size=13),
        title=dict(
            text=f"<b>{title}</b><br>"
                 f"<span style='font-size:12px;color:{PALETTE['sub']}'>{subtitle}</span>",
            font=dict(family=FONT, size=17, color=PALETTE['navy']),
            x=0.04, y=0.97, xanchor='left',
        ),
        margin=dict(l=75, r=45, t=95, b=75),
        hoverlabel=dict(
            bgcolor=PALETTE['navy'], font_color='white',
            font_family=FONT, font_size=12,
            bordercolor=PALETTE['navy'],
        ),
    )
 
AXIS_BASE = dict(
    gridcolor=PALETTE['grid'], gridwidth=1,
    linecolor=PALETTE['grid'], zeroline=False,
    tickfont=dict(family=FONT, size=12, color=PALETTE['sub']),
    title_font=dict(family=FONT, size=13, color=PALETTE['navy']),
)

# VISUALIZATION 1: Readmission Rate by Age Group

age_order = ['[0-10)','[10-20)','[20-30)','[30-40)','[40-50)',
             '[50-60)','[60-70)','[70-80)','[80-90)','[90-100)']
 
readmit_by_age = (
    df.groupby('age')['readmitted_binary']
    .mean().reset_index()
    .rename(columns={'readmitted_binary': 'readmit_rate'})
)
readmit_by_age['age'] = pd.Categorical(
    readmit_by_age['age'], categories=age_order, ordered=True
)
readmit_by_age = readmit_by_age.sort_values('age')
ages  = readmit_by_age['age'].astype(str).tolist()
rates = readmit_by_age['readmit_rate'].tolist()
peak_idx = int(np.argmax(rates))
colors1  = coral_gradient(rates)
 
fig1 = go.Figure()
 
# Main bars
fig1.add_trace(go.Bar(
    x=ages, y=rates,
    marker=dict(
        color=colors1,
        line=dict(color=PALETTE['panel'], width=1.8),
        cornerradius=5,
    ),
    text=[f"{r:.1%}" for r in rates],
    textposition='outside',
    textfont=dict(family=FONT, size=11, color=PALETTE['navy']),
    hovertemplate='<b>Age Group:</b> %{x}<br><b>Rate:</b> %{y:.2%}<extra></extra>',
    name='',
))
 
# Mean reference line
mean_rate = float(np.mean(rates))
fig1.add_hline(
    y=mean_rate,
    line=dict(color=PALETTE['sub'], width=1.4, dash='dot'),
    annotation_text=f"Overall avg {mean_rate:.1%}",
    annotation_position="top right",
    annotation_font=dict(family=FONT, size=11, color=PALETTE['sub']),
)
 
# Peak callout
fig1.add_annotation(
    x=ages[peak_idx], y=rates[peak_idx],
    text=f"Peak ▲<br>{rates[peak_idx]:.1%}",
    showarrow=True, arrowhead=2, arrowcolor=PALETTE['coral'],
    arrowwidth=1.5, ax=40, ay=-40,
    font=dict(family=FONT, size=11, color=PALETTE['coral']),
    bgcolor='#FFF0EE', bordercolor=PALETTE['coral'], borderwidth=1,
    borderpad=4,
)
 
fig1.update_layout(
    **base_layout(
        'Early Readmission Rate by Age Group',
        'Proportion of patients readmitted within 30 days - by age bracket'
    ),
    xaxis_tickangle=-30,
    yaxis_tickformat='.0%',
    yaxis_range=[0, max(rates)*1.35],
    showlegend=False,
)
fig1.update_xaxes(**AXIS_BASE, showgrid=True, title_text='Age Group')
fig1.update_yaxes(**AXIS_BASE, showgrid=True, title_text='30-Day Readmission Rate')
fig1.show()
 
# 💡 INSIGHT: Middle-aged groups (40–60) show higher early readmission than
# the elderly — older patients often receive more intensive post-discharge care.
# Age is an important ML feature.
 

# VISUALIZATION 2: Medications by Readmission Status
cat_order  = ['NO',        '>30',               '<30']
cat_labels = ['Not Readmitted', 'Readmitted >30d', 'Readmitted <30d']
cat_colors = [PALETTE['steel'],  PALETTE['amber'],   PALETTE['coral']]
 
fig2 = go.Figure()
for cat, label, color in zip(cat_order, cat_labels, cat_colors):
    subset = df[df['readmitted'] == cat]['num_medications'].dropna()
    q1, med, q3 = subset.quantile([0.25, 0.5, 0.75])
    fig2.add_trace(go.Box(
        y=subset,
        name=label,
        marker=dict(color=color, size=3, opacity=0.45, outliercolor=color),
        line=dict(color=color, width=2.2),
        fillcolor='rgba({},{},{},0.12)'.format(
            int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
        ),
        boxmean='sd',
        whiskerwidth=0.55,
        hovertemplate=(
            f'<b>{label}</b><br>'
            'Median: %{median:.0f}<br>'
            'IQR: %{q1:.0f} – %{q3:.0f}<br>'
            'Whiskers: %{lowerfence:.0f} – %{upperfence:.0f}'
            '<extra></extra>'
        ),
    ))
 
fig2.update_layout(
    **base_layout(
        'Medication Count by Readmission Status',
        'Higher medication burden associated with earlier readmission - proxy for disease complexity'
    ),
    showlegend=True,
    legend=dict(
        orientation='h', yanchor='bottom', y=1.02,
        xanchor='right', x=1,
        font=dict(family=FONT, size=12),
        bgcolor='rgba(0,0,0,0)',
    ),
    boxmode='group',
    boxgap=0.35,
)
fig2.update_xaxes(**AXIS_BASE, showgrid=False, title_text='Readmission Category')
fig2.update_yaxes(**AXIS_BASE, showgrid=True,  title_text='Number of Medications')
fig2.show()
 
# 💡 INSIGHT: Patients readmitted <30 days carry a higher median medication count
# (~18 vs ~15), confirming num_medications as a strong ML predictor.

# VISUALIZATION 3: Insulin Usage vs Readmission
insulin_order  = ['No', 'Steady', 'Up', 'Down']
readmit_order  = ['NO', '>30', '<30']
readmit_labels = {
    'NO':  'Not Readmitted',
    '>30': 'Readmitted >30d',
    '<30': 'Readmitted <30d',
}
readmit_colors = {
    'NO':  PALETTE['steel'],
    '>30': PALETTE['amber'],
    '<30': PALETTE['coral'],
}
 
insulin_readmit = (
    df.groupby(['insulin','readmitted']).size()
    .reset_index(name='count')
)
insulin_readmit['pct'] = insulin_readmit.groupby('insulin')['count'].transform(
    lambda x: x / x.sum() * 100
)
 
fig3 = go.Figure()
for cat in readmit_order:
    sub = (
        insulin_readmit[insulin_readmit['readmitted']==cat]
        .set_index('insulin').reindex(insulin_order).reset_index()
    )
    fig3.add_trace(go.Bar(
        name=readmit_labels[cat],
        x=sub['insulin'],
        y=sub['pct'],
        marker=dict(
            color=readmit_colors[cat],
            line=dict(color=PALETTE['panel'], width=1.3),
            cornerradius=4,
        ),
        text=sub['pct'].apply(lambda v: f"{v:.1f}%" if pd.notna(v) else ""),
        textposition='outside',
        textfont=dict(family=FONT, size=10, color=PALETTE['navy']),
        hovertemplate=(
            f'<b>{readmit_labels[cat]}</b><br>'
            'Insulin: %{x}<br>Share: %{y:.1f}%<extra></extra>'
        ),
    ))
 
fig3.update_layout(
    **base_layout(
        'Readmission Outcome by Insulin Usage',
        'Upward insulin adjustment correlates with higher early readmission proportion'
    ),
    barmode='group',
    bargap=0.22,
    bargroupgap=0.06,
    showlegend=True,
    legend=dict(
        orientation='h', yanchor='bottom', y=1.02,
        xanchor='right', x=1,
        font=dict(family=FONT, size=12),
        bgcolor='rgba(0,0,0,0)',
    ),
    yaxis_range=[0, insulin_readmit['pct'].max()*1.28],
)
fig3.update_xaxes(**AXIS_BASE, showgrid=False, title_text='Insulin Status')
fig3.update_yaxes(**AXIS_BASE, showgrid=True,  title_text='% of Patients within Group')
fig3.show()
 
# 💡 INSIGHT: "Up" insulin changes show higher <30d readmission — signals
# worsening glycaemic control. Insulin status included as ML feature.
 

dashboard = make_subplots(
    rows=1, cols=3,
    subplot_titles=[
        '<b>Fig 1 - Readmission Rate by Age Group</b>',
        '<b>Fig 2 - Medications by Readmission Status</b>',
        '<b>Fig 3 - Readmission Outcome by Insulin Usage</b>',
    ],
    column_widths=[0.36, 0.28, 0.36],
    horizontal_spacing=0.09,
    specs=[[{"type":"bar"}, {"type":"box"}, {"type":"bar"}]],
)
 
# Fig 1: Age bars 
dashboard.add_trace(go.Bar(
    x=ages, y=rates,
    marker=dict(color=colors1, line=dict(color='white', width=1), cornerradius=4),
    text=[f"{r:.1%}" for r in rates],
    textposition='outside',
    textfont=dict(family=FONT, size=8),
    hovertemplate='Age: %{x}<br>Rate: %{y:.2%}<extra></extra>',
    showlegend=False,
), row=1, col=1)
 
dashboard.add_hline(
    y=mean_rate, line=dict(color=PALETTE['sub'], width=1.2, dash='dot'),
    row=1, col=1,
)
 
# Fig 2: Medications boxes 
for cat, label, color in zip(cat_order, cat_labels, cat_colors):
    subset = df[df['readmitted']==cat]['num_medications'].dropna()
    dashboard.add_trace(go.Box(
        y=subset, name=label,
        marker=dict(color=color, size=2, opacity=0.35),
        line=dict(color=color, width=1.8),
        fillcolor='rgba({},{},{},0.09)'.format(
            int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
        ),
        boxmean=True,
        showlegend=True,
        legendgroup=label,
        hovertemplate=f'<b>{label}</b><br>Median: %{{median:.0f}}<extra></extra>',
    ), row=1, col=2)
 
# Fig 3: Insulin grouped bars 
for cat in readmit_order:
    sub = (
        insulin_readmit[insulin_readmit['readmitted']==cat]
        .set_index('insulin').reindex(insulin_order).reset_index()
    )
    dashboard.add_trace(go.Bar(
        name=readmit_labels[cat],
        x=sub['insulin'], y=sub['pct'],
        marker=dict(color=readmit_colors[cat],
                    line=dict(color='white', width=1), cornerradius=3),
        text=sub['pct'].apply(lambda v: f"{v:.1f}%" if pd.notna(v) else ""),
        textposition='outside',
        textfont=dict(family=FONT, size=8),
        showlegend=False,
        hovertemplate=f'<b>{readmit_labels[cat]}</b><br>%{{y:.1f}}%<extra></extra>',
    ), row=1, col=3)
 
# Dashboard layout
dashboard.update_layout(
    height=560,
    plot_bgcolor=PALETTE['panel'],
    paper_bgcolor=PALETTE['bg'],
    font=dict(family=FONT, color=PALETTE['navy'], size=12),
    title=dict(
        text=(
            "<b>Diabetes Hospital Readmission - EDA Dashboard</b><br>"
            f"<span style='font-size:13px;color:{PALETTE['sub']}'>"
            "UCI 130-US Hospitals Dataset  |  3 Exploratory Visualizations</span>"
        ),
        font=dict(family=FONT, size=20, color=PALETTE['navy']),
        x=0.04, y=0.98, xanchor='left',
    ),
    margin=dict(l=65, r=40, t=110, b=90),
    legend=dict(
        orientation='h',
        yanchor='bottom', y=-0.18,
        xanchor='center', x=0.5,
        font=dict(family=FONT, size=11),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor=PALETTE['grid'],
        borderwidth=1,
    ),
    hoverlabel=dict(
        bgcolor=PALETTE['navy'], font_color='white',
        font_family=FONT, font_size=11,
    ),
    boxmode='group',
    barmode='group',
)
 
# Per-subplot axes
subplot_axes = {
    'xaxis':  dict(title_text='Age Group',            tickangle=-35,    showgrid=True),
    'yaxis':  dict(title_text='Readmission Rate',     tickformat='.0%', showgrid=True),
    'xaxis2': dict(title_text='Readmission Category',                   showgrid=False),
    'yaxis2': dict(title_text='No. of Medications',                     showgrid=True),
    'xaxis3': dict(title_text='Insulin Status',                         showgrid=False),
    'yaxis3': dict(title_text='% of Patients',                          showgrid=True),
}
for key, cfg in subplot_axes.items():
    full_cfg = {
        **AXIS_BASE, **cfg,
        'tickfont':   dict(family=FONT, size=10, color=PALETTE['sub']),
        'title_font': dict(family=FONT, size=11, color=PALETTE['navy']),
    }
    dashboard.update_layout({key: full_cfg})
 
# Subplot title font
for ann in dashboard.layout.annotations:
    ann.font.update(family=FONT, size=12, color=PALETTE['navy'])
 
dashboard.show()
 
 
# Save dashboard as HTML file 
dashboard.write_html('eda_dashboard.html')
files.download('eda_dashboard.html')
print("Downloaded: eda_dashboard.html")
