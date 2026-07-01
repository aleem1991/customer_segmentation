import React from 'react';

interface RiskDistributionChartProps {
  high: number;
  med: number;
  low: number;
  activeFilter: string | null;
  onFilterChange: (filter: string | null) => void;
}

export const RiskDistributionChart: React.FC<RiskDistributionChartProps> = ({
  high,
  med,
  low,
  activeFilter,
  onFilterChange
}) => {
  const total = high + med + low;
  
  const r = 50;
  const circ = 2 * Math.PI * r; // ~314.16
  
  const highPct = total ? (high / total) : 0;
  const medPct = total ? (med / total) : 0;
  const lowPct = total ? (low / total) : 0;
  
  const highDash = highPct * circ;
  const medDash = medPct * circ;
  const lowDash = lowPct * circ;
  
  const highOffset = 0;
  const medOffset = -highDash;
  const lowOffset = -(highDash + medDash);

  const handleSliceClick = (tier: string) => {
    if (activeFilter === tier) {
      onFilterChange(null);
    } else {
      onFilterChange(tier);
    }
  };

  return (
    <div className="card chart-card">
      <div className="chart-header">
        <span className="chart-title">Churn Risk Distribution</span>
        {activeFilter && (
          <button 
            className="btn" 
            style={{ padding: '4px 8px', fontSize: '11px' }}
            onClick={() => onFilterChange(null)}
          >
            Clear Filter
          </button>
        )}
      </div>

      <div className="chart-content" style={{ display: 'flex', justifyContent: 'space-around', alignItems: 'center' }}>
        <div className="svg-chart-container" style={{ width: '160px', height: '160px' }}>
          <svg width="100%" height="100%" viewBox="0 0 140 140">
            {/* Background circle */}
            <circle 
              cx="70" 
              cy="70" 
              r={r} 
              fill="transparent" 
              stroke="var(--border-color)" 
              strokeWidth="10" 
            />
            
            {/* Low Risk Segment */}
            {low > 0 && (
              <circle 
                cx="70" 
                cy="70" 
                r={r} 
                fill="transparent" 
                stroke="var(--color-low)" 
                strokeWidth={activeFilter === 'Low Risk' ? "15" : "10"} 
                strokeDasharray={`${lowDash} ${circ}`} 
                strokeDashoffset={lowOffset} 
                transform="rotate(-90 70 70)"
                className="svg-donut-slice"
                style={{ strokeLinecap: 'round', opacity: activeFilter && activeFilter !== 'Low Risk' ? 0.3 : 1 }}
                onClick={() => handleSliceClick('Low Risk')}
              />
            )}
            
            {/* Medium Risk Segment */}
            {med > 0 && (
              <circle 
                cx="70" 
                cy="70" 
                r={r} 
                fill="transparent" 
                stroke="var(--color-med)" 
                strokeWidth={activeFilter === 'Medium Risk' ? "15" : "10"} 
                strokeDasharray={`${medDash} ${circ}`} 
                strokeDashoffset={medOffset} 
                transform="rotate(-90 70 70)"
                className="svg-donut-slice"
                style={{ strokeLinecap: 'round', opacity: activeFilter && activeFilter !== 'Medium Risk' ? 0.3 : 1 }}
                onClick={() => handleSliceClick('Medium Risk')}
              />
            )}

            {/* High Risk Segment */}
            {high > 0 && (
              <circle 
                cx="70" 
                cy="70" 
                r={r} 
                fill="transparent" 
                stroke="var(--color-high)" 
                strokeWidth={activeFilter === 'High Risk' ? "15" : "10"} 
                strokeDasharray={`${highDash} ${circ}`} 
                strokeDashoffset={highOffset} 
                transform="rotate(-90 70 70)"
                className="svg-donut-slice"
                style={{ strokeLinecap: 'round', opacity: activeFilter && activeFilter !== 'High Risk' ? 0.3 : 1 }}
                onClick={() => handleSliceClick('High Risk')}
              />
            )}

            {/* Inner Label */}
            <text x="70" y="65" textAnchor="middle" fill="var(--text-secondary)" fontSize="10px" fontWeight="600">
              TOTAL
            </text>
            <text x="70" y="85" textAnchor="middle" fill="var(--text-primary)" fontSize="18px" fontWeight="800" fontFamily="var(--font-heading)">
              {total.toLocaleString()}
            </text>
          </svg>
        </div>

        <div className="chart-legend" style={{ margin: '0' }}>
          <div 
            className={`legend-item ${activeFilter === 'High Risk' ? 'active' : ''}`}
            onClick={() => handleSliceClick('High Risk')}
            style={{ borderLeft: activeFilter === 'High Risk' ? '2px solid var(--color-high)' : 'none', paddingLeft: '8px' }}
          >
            <div className="legend-color" style={{ backgroundColor: 'var(--color-high)' }} />
            <span>High Risk ({high} | {(highPct * 100).toFixed(0)}%)</span>
          </div>
          
          <div 
            className={`legend-item ${activeFilter === 'Medium Risk' ? 'active' : ''}`}
            onClick={() => handleSliceClick('Medium Risk')}
            style={{ borderLeft: activeFilter === 'Medium Risk' ? '2px solid var(--color-med)' : 'none', paddingLeft: '8px' }}
          >
            <div className="legend-color" style={{ backgroundColor: 'var(--color-med)' }} />
            <span>Medium Risk ({med} | {(medPct * 100).toFixed(0)}%)</span>
          </div>

          <div 
            className={`legend-item ${activeFilter === 'Low Risk' ? 'active' : ''}`}
            onClick={() => handleSliceClick('Low Risk')}
            style={{ borderLeft: activeFilter === 'Low Risk' ? '2px solid var(--color-low)' : 'none', paddingLeft: '8px' }}
          >
            <div className="legend-color" style={{ backgroundColor: 'var(--color-low)' }} />
            <span>Low Risk ({low} | {(lowPct * 100).toFixed(0)}%)</span>
          </div>
        </div>
      </div>
    </div>
  );
};
