import React from 'react';
import { 
  User, Clock, ShoppingCart, DollarSign, 
  ShoppingBag, AlertTriangle, CheckCircle2, Info 
} from 'lucide-react';

interface CustomerRecord {
  id: string;
  recency: number;
  frequency: number;
  monetary: number;
  basketSize: number;
  churnProb: number;
  riskTier: string;
  segment: string;
  recommendation: string;
  shapValues?: {
    [key: string]: number;
  };
}

interface CustomerDetailsProps {
  customer: CustomerRecord | null;
}

export const CustomerDetails: React.FC<CustomerDetailsProps> = ({ customer }) => {
  if (!customer) {
    return (
      <div className="card details-card" style={{ justifyContent: 'center', minHeight: '400px' }}>
        <div className="empty-state">
          <User size={48} className="text-muted" />
          <h3>No Customer Selected</h3>
          <p>Click a customer row in the table or search for a Customer ID to view their detailed churn profile.</p>
        </div>
      </div>
    );
  }

  const isHighRisk = customer.riskTier === 'High Risk';
  const isMedRisk = customer.riskTier === 'Medium Risk';
  
  const riskColor = isHighRisk 
    ? 'var(--color-high)' 
    : isMedRisk 
      ? 'var(--color-med)' 
      : 'var(--color-low)';

  const riskClass = isHighRisk 
    ? 'high-risk' 
    : isMedRisk 
      ? 'med-risk' 
      : 'low-risk';

  const riskIcon = isHighRisk 
    ? <AlertTriangle className="recommendation-icon high" size={20} />
    : isMedRisk
      ? <AlertTriangle className="recommendation-icon med" size={20} />
      : <CheckCircle2 className="recommendation-icon low" size={20} />;

  return (
    <div className="card details-card">
      <div className="details-header">
        <div>
          <div className="details-customer-id">ID: {customer.id}</div>
          <div className="details-segment">Segment: <strong>{customer.segment}</strong></div>
        </div>
        <span className={`badge badge-${isHighRisk ? 'high' : isMedRisk ? 'med' : 'low'}`}>
          {customer.riskTier}
        </span>
      </div>

      <div className="risk-gauge-container">
        <div className="risk-gauge-header">
          <span>Predicted Churn Probability</span>
          <span style={{ color: riskColor }}>{(customer.churnProb * 100).toFixed(1)}%</span>
        </div>
        <div className="risk-gauge-track">
          <div 
            className="risk-gauge-fill" 
            style={{ 
              width: `${customer.churnProb * 100}%`,
              backgroundColor: riskColor,
              boxShadow: `0 0 8px ${riskColor}50`
            }}
          />
        </div>
      </div>

      <div className="metrics-grid">
        <div className="metric-item">
          <div className="metric-label" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Clock size={12} /> Recency
          </div>
          <div className="metric-val">{customer.recency} Days</div>
          <div className="kpi-subtext">Since last order</div>
        </div>
        
        <div className="metric-item">
          <div className="metric-label" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <ShoppingCart size={12} /> Frequency
          </div>
          <div className="metric-val">{customer.frequency} Orders</div>
          <div className="kpi-subtext">Total invoices</div>
        </div>

        <div className="metric-item">
          <div className="metric-label" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <DollarSign size={12} /> Mean Monetary
          </div>
          <div className="metric-val">${customer.monetary.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</div>
          <div className="kpi-subtext">Average spend per invoice</div>
        </div>

        <div className="metric-item">
          <div className="metric-label" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <ShoppingBag size={12} /> Avg Basket Size
          </div>
          <div className="metric-val">{customer.basketSize.toFixed(1)} items</div>
          <div className="kpi-subtext">Average items per order</div>
        </div>
      </div>

      {customer.shapValues && (
        <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '16px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '4px' }}>
            Model Decision Drivers (TreeSHAP Impact)
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {Object.entries(customer.shapValues)
              .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
              .slice(0, 5) // Show top 5 drivers
              .map(([feature, val]) => {
                const isPositive = val > 0;
                const absVal = Math.abs(val);
                const widthPct = Math.min(50, (absVal / 0.8) * 50);
                
                return (
                  <div key={feature} style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px' }}>
                      <span style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>{feature}</span>
                      <span style={{ fontWeight: 600, color: isPositive ? 'var(--color-high)' : 'var(--color-low)' }}>
                        {isPositive ? '+' : ''}{val.toFixed(3)}
                      </span>
                    </div>
                    {/* Double-sided bar chart */}
                    <div style={{ display: 'flex', width: '100%', height: '8px', backgroundColor: 'var(--bg-subtle)', borderRadius: '4px', position: 'relative', overflow: 'hidden' }}>
                      {/* Midline separator */}
                      <div style={{ position: 'absolute', left: '50%', top: '0', bottom: '0', width: '1px', backgroundColor: 'var(--border-color)', zIndex: 1 }} />
                      
                      {/* Bar Fill */}
                      <div style={{ 
                        height: '100%', 
                        width: `${widthPct}%`, 
                        backgroundColor: isPositive ? 'var(--color-high)' : 'var(--color-low)', 
                        position: 'absolute',
                        left: isPositive ? '50%' : 'auto',
                        right: isPositive ? 'auto' : '50%',
                        borderRadius: isPositive ? '0 4px 4px 0' : '4px 0 0 4px',
                        transition: 'all 0.4s ease-out'
                      }} />
                    </div>
                  </div>
                );
              })}
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9px', color: 'var(--text-muted)', marginTop: '2px' }}>
              <span>◀ Reduces Churn Risk (Loyal)</span>
              <span>Increases Churn Risk (Leaving) ▶</span>
            </div>
          </div>
        </div>
      )}

      <div className={`recommendation-box ${riskClass}`}>
        {riskIcon}
        <div className="recommendation-content">
          <h4>Actionable Recommendation</h4>
          <p>{customer.recommendation}</p>
        </div>
      </div>
      
      <div style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
        <Info size={12} /> Churn predictions are generated using XGBoost based on rolling transaction windows.
      </div>
    </div>
  );
};
