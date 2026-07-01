import React from 'react';

interface MetricCardProps {
  title: string;
  value: string | number;
  subtext: string;
  icon: React.ReactNode;
  iconColor: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtext,
  icon,
  iconColor
}) => {
  return (
    <div className="card">
      <div className="kpi-card-header">
        <span className="kpi-title">{title}</span>
        <div className="kpi-icon" style={{ backgroundColor: `color-mix(in srgb, ${iconColor} 8%, transparent)`, color: iconColor }}>
          {icon}
        </div>
      </div>
      <div className="kpi-value">{value}</div>
      <div className="kpi-subtext">{subtext}</div>
    </div>
  );
};
