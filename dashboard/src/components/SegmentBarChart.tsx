import React from 'react';


interface SegmentBarChartProps {
  segments: { [key: string]: number };
  activeFilter: string | null;
  onFilterChange: (filter: string | null) => void;
}

export const SegmentBarChart: React.FC<SegmentBarChartProps> = ({
  segments,
  activeFilter,
  onFilterChange
}) => {
  const segmentDataList = Object.entries(segments)
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count);

  const maxCount = segmentDataList.length > 0 ? Math.max(...segmentDataList.map(s => s.count)) : 1;

  const getSegmentColor = (name: string) => {
    const lowercaseName = name.toLowerCase();
    if (lowercaseName.includes('champion')) return 'var(--color-low)'; // emerald green
    if (lowercaseName.includes('loyalist')) return 'var(--color-blue)'; // blue
    if (lowercaseName.includes('promising')) return '#818cf8'; // indigo
    if (lowercaseName.includes('hibernating')) return 'var(--color-high)'; // rose red
    if (lowercaseName.includes('sleep') || lowercaseName.includes('risk')) return 'var(--color-med)'; // amber orange
    return 'var(--text-secondary)'; // grey
  };

  const handleRowClick = (name: string) => {
    if (activeFilter === name) {
      onFilterChange(null);
    } else {
      onFilterChange(name);
    }
  };

  return (
    <div className="card chart-card">
      <div className="chart-header">
        <span className="chart-title">Customer Segmentation</span>
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

      <div className="chart-content" style={{ display: 'block', padding: '0 8px' }}>
        <div className="bar-chart-container">
          {segmentDataList.length === 0 ? (
            <div style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '40px' }}>
              No segmentation data available.
            </div>
          ) : (
            segmentDataList.map(({ name, count }) => {
              const percentage = (count / maxCount) * 100;
              const color = getSegmentColor(name);
              const isActive = activeFilter === name;
              
              return (
                <div 
                  key={name}
                  className={`bar-row ${isActive ? 'active' : ''}`}
                  onClick={() => handleRowClick(name)}
                >
                  <span className="bar-label" title={name}>{name}</span>
                  <div className="bar-track">
                    <div 
                      className="bar-fill" 
                      style={{ 
                        width: `${percentage}%`, 
                        backgroundColor: color,
                        boxShadow: isActive ? `0 0 8px ${color}60` : 'none'
                      }}
                    />
                  </div>
                  <span className="bar-value">{count.toLocaleString()}</span>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
};
