import { useState, useEffect, useRef } from 'react';
import { 
  Users, DollarSign, TrendingUp, AlertTriangle, 
  Search, ShieldAlert, Sparkles, RefreshCw, Sun, Moon,
  Calculator, Play, Activity, CheckCircle
} from 'lucide-react';
import { MetricCard } from './components/MetricCard';
import { RiskDistributionChart } from './components/RiskDistributionChart';
import { SegmentBarChart } from './components/SegmentBarChart';
import { CustomerTable } from './components/CustomerTable';
import { CustomerDetails } from './components/CustomerDetails';

interface SummaryData {
  total_customers: number;
  revenue_at_risk: number;
  risk_tiers: {
    "High Risk": number;
    "Medium Risk": number;
    "Low Risk": number;
  };
  segment_distribution: {
    [key: string]: number;
  };
  segment_churn: {
    [key: string]: {
      total: number;
      high_risk: number;
      high_risk_pct: number;
    };
  };
}

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
}

function App() {
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    const saved = localStorage.getItem('theme');
    return (saved as 'dark' | 'light') || 'dark';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const [activeTab, setActiveTab] = useState<'insights' | 'calculator' | 'monitor'>('insights');

  // Drift Monitoring States
  const [driftData, setDriftData] = useState<any>(null);
  const [driftLoading, setDriftLoading] = useState<boolean>(false);
  const [driftError, setDriftError] = useState<string | null>(null);

  // Shadow Deployment States
  const [shadowData, setShadowData] = useState<any>(null);
  const [shadowLoading, setShadowLoading] = useState<boolean>(false);
  const [shadowError, setShadowError] = useState<string | null>(null);

  // WebSocket Live Streaming States
  const [liveLogs, setLiveLogs] = useState<any[]>([]);
  const [wsStatus, setWsStatus] = useState<'connected' | 'disconnected' | 'connecting'>('disconnected');
  const wsRef = useRef<WebSocket | null>(null);

  // Calculator Form States
  const [calcRecency, setCalcRecency] = useState<number>(65);
  const [calcFrequency, setCalcFrequency] = useState<number>(3);
  const [calcMonetary, setCalcMonetary] = useState<number>(350.5);
  const [calcBasketSize, setCalcBasketSize] = useState<number>(12.5);
  const [calcAvgDaysBetween, setCalcAvgDaysBetween] = useState<string>('30');
  const [calcRecentOrdersRatio, setCalcRecentOrdersRatio] = useState<string>('1');
  const [calcIsUk, setCalcIsUk] = useState<number>(1);

  // Calculator Prediction Output States
  const [calcResult, setCalcResult] = useState<{
    churn_probability: number;
    risk_tier: string;
    recommendation: string;
    shap_values?: {
      [key: string]: number;
    };
  } | null>(null);
  const [calcLoading, setCalcLoading] = useState<boolean>(false);
  const [calcError, setCalcError] = useState<string | null>(null);

  const [summary, setSummary] = useState<SummaryData | null>(null);
  const [customers, setCustomers] = useState<CustomerRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filter and Search States
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCustomer, setSelectedCustomer] = useState<CustomerRecord | null>(null);
  const [segmentFilter, setSegmentFilter] = useState<string | null>(null);
  const [riskFilter, setRiskFilter] = useState<string | null>(null);
  const [showSearchResults, setShowSearchResults] = useState(false);

  // Sorting and Pagination
  const [sortField, setSortField] = useState<string>('churnProb');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 10;
  
  const searchRef = useRef<HTMLDivElement>(null);

  // Fetch Data
  const loadData = () => {
    setLoading(true);
    setError(null);
    Promise.all([
      fetch('/data/summary.json').then(res => {
        if (!res.ok) throw new Error("Failed to load summary stats");
        return res.json();
      }),
      fetch('/data/customers.json').then(res => {
        if (!res.ok) throw new Error("Failed to load customer details");
        return res.json();
      })
    ])
      .then(([summaryData, customersData]) => {
        setSummary(summaryData);
        setCustomers(customersData);
        
        // Default select the first customer (prefer high risk)
        const firstHighRisk = customersData.find((c: any) => c.riskTier === 'High Risk');
        setSelectedCustomer(firstHighRisk || customersData[0] || null);
        setLoading(false);
      })
      .catch(err => {
        console.error("Dashboard Loading Error:", err);
        setError("Make sure you have run 'python src/export_json.py' to generate the data files.");
        setLoading(false);
      });
  };

  const runLiveForecast = () => {
    setCalcLoading(true);
    setCalcError(null);
    
    const payload = {
      recency: Number(calcRecency) || 0,
      frequency: Number(calcFrequency) || 1,
      monetary: Number(calcMonetary) || 0.0,
      basket_size: Number(calcBasketSize) || 0.0,
      avg_days_between: calcAvgDaysBetween.trim() !== '' ? Number(calcAvgDaysBetween) : null,
      recent_orders_ratio: calcRecentOrdersRatio.trim() !== '' ? Number(calcRecentOrdersRatio) : null,
      is_uk: Number(calcIsUk)
    };

    fetch('http://127.0.0.1:8000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
      .then(res => {
        if (!res.ok) throw new Error("FastAPI model service is offline. Please make sure the API server is running on port 8000.");
        return res.json();
      })
      .then(data => {
        setCalcResult(data);
        setCalcLoading(false);
      })
      .catch(err => {
        console.error("Live Forecast Error:", err);
        setCalcError(err.message || "Failed to contact prediction server.");
        setCalcLoading(false);
      });
  };

  const checkModelDrift = () => {
    setDriftLoading(true);
    setDriftError(null);
    fetch('http://127.0.0.1:8000/monitor')
      .then(res => {
        if (!res.ok) throw new Error("FastAPI model service is offline. Please make sure the API server is running on port 8000.");
        return res.json();
      })
      .then(data => {
        setDriftData(data);
        setDriftLoading(false);
      })
      .catch(err => {
        console.error("Drift Monitoring Error:", err);
        setDriftError(err.message || "Failed to fetch drift metrics.");
        setDriftLoading(false);
      });
  };

  const fetchShadowStats = () => {
    setShadowLoading(true);
    setShadowError(null);
    fetch('http://127.0.0.1:8000/shadow_stats')
      .then(res => {
        if (!res.ok) throw new Error("FastAPI model service is offline. Please make sure the API server is running on port 8000.");
        return res.json();
      })
      .then(data => {
        setShadowData(data);
        setShadowLoading(false);
      })
      .catch(err => {
        console.error("Shadow stats load failed:", err);
        setShadowError(err.message || "Failed to load shadow deployment metrics.");
        setShadowLoading(false);
      });
  };

  const toggleWebSocket = () => {
    if (wsStatus === 'connected' || wsRef.current) {
      if (wsRef.current) {
        wsRef.current.close();
      }
      setWsStatus('disconnected');
      return;
    }

    setWsStatus('connecting');
    const ws = new WebSocket('ws://127.0.0.1:8000/ws/transactions');
    wsRef.current = ws;

    ws.onopen = () => {
      setWsStatus('connected');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'TRANSACTION') {
          // Append to log events
          setLiveLogs(prev => [data, ...prev].slice(0, 8));
          
          // Update customer in list
          setCustomers(prev => prev.map(c => {
            if (c.id.toString() === data.id.toString()) {
              return {
                ...c,
                recency: data.newMetrics.recency,
                frequency: data.newMetrics.frequency,
                monetary: data.newMetrics.monetary,
                basketSize: data.newMetrics.basketSize,
                churnProb: data.newMetrics.churnProb,
                riskTier: data.newMetrics.riskTier
              };
            }
            return c;
          }));

          // Update current details view if applicable
          setSelectedCustomer(prev => {
            if (prev && prev.id.toString() === data.id.toString()) {
              return {
                ...prev,
                recency: data.newMetrics.recency,
                frequency: data.newMetrics.frequency,
                monetary: data.newMetrics.monetary,
                basketSize: data.newMetrics.basketSize,
                churnProb: data.newMetrics.churnProb,
                riskTier: data.newMetrics.riskTier
              };
            }
            return prev;
          });
        }
      } catch (err) {
        console.error("Failed to parse WS stream data:", err);
      }
    };

    ws.onerror = (err) => {
      console.error("WebSocket connection error:", err);
      setWsStatus('disconnected');
    };

    ws.onclose = () => {
      setWsStatus('disconnected');
      wsRef.current = null;
    };
  };

  const injectDriftData = async () => {
    setDriftLoading(true);
    setDriftError(null);
    try {
      const profiles = Array.from({ length: 15 }, () => ({
        recency: Math.floor(Math.random() * 200) + 150,
        frequency: Math.floor(Math.random() * 2) + 1,
        monetary: Math.random() * 4000 + 1500,
        basket_size: Math.random() * 50 + 30,
        avg_days_between: null,
        recent_orders_ratio: null,
        is_uk: 1
      }));

      await Promise.all(
        profiles.map(profile => 
          fetch('http://127.0.0.1:8000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(profile)
          })
        )
      );

      checkModelDrift();
      fetchShadowStats();
    } catch (err: any) {
      console.error("Mock Drift Injection Failed:", err);
      setDriftError(err.message || "Failed to inject mock drift data.");
      setDriftLoading(false);
    }
  };

  useEffect(() => {
    loadData();
    
    // Close search dropdown on click outside
    const handleClickOutside = (event: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(event.target as Node)) {
        setShowSearchResults(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  useEffect(() => {
    if (activeTab === 'monitor') {
      checkModelDrift();
      fetchShadowStats();
    }
  }, [activeTab]);

  // Filter & Sort Logic
  const filteredCustomers = customers.filter(customer => {
    const matchesSegment = !segmentFilter || customer.segment === segmentFilter;
    const matchesRisk = !riskFilter || customer.riskTier === riskFilter;
    const matchesSearch = !searchQuery || customer.id.includes(searchQuery);
    return matchesSegment && matchesRisk && matchesSearch;
  });

  const sortedCustomers = [...filteredCustomers].sort((a: any, b: any) => {
    let aVal = a[sortField];
    let bVal = b[sortField];

    if (sortField === 'id') {
      const aNum = parseInt(a.id) || 0;
      const bNum = parseInt(b.id) || 0;
      return sortDirection === 'asc' ? aNum - bNum : bNum - aNum;
    }

    if (typeof aVal === 'string') {
      return sortDirection === 'asc' 
        ? aVal.localeCompare(bVal) 
        : bVal.localeCompare(aVal);
    }

    return sortDirection === 'asc' 
      ? aVal - bVal 
      : bVal - aVal;
  });

  // Handle pagination resets on filter changes
  const handleSegmentFilterChange = (filter: string | null) => {
    setSegmentFilter(filter);
    setCurrentPage(1);
  };

  const handleRiskFilterChange = (filter: string | null) => {
    setRiskFilter(filter);
    setCurrentPage(1);
  };

  const handleSort = (field: string) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
    setCurrentPage(1);
  };

  // Search autocomplete list
  const searchSuggestions = searchQuery.trim() !== ''
    ? customers.filter(c => c.id.includes(searchQuery)).slice(0, 5)
    : [];

  const handleSelectSuggestion = (customer: CustomerRecord) => {
    setSelectedCustomer(customer);
    setSearchQuery(customer.id);
    setShowSearchResults(false);
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '80vh', gap: '16px' }}>
        <RefreshCw className="logo-icon" size={48} style={{ animation: 'spin 2s linear infinite' }} />
        <p style={{ color: 'var(--text-secondary)' }}>Loading customer prediction database...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '80vh', gap: '20px', textAlign: 'center', padding: '24px' }}>
        <ShieldAlert size={64} style={{ color: 'var(--color-high)' }} />
        <h2 style={{ fontFamily: 'var(--font-heading)' }}>Data Files Missing</h2>
        <p style={{ color: 'var(--text-secondary)', maxWidth: '450px' }}>
          {error}
        </p>
        <button className="btn" onClick={loadData} style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '10px 20px', borderColor: 'var(--color-blue)', color: 'var(--text-primary)' }}>
          <RefreshCw size={16} /> Retry Connection
        </button>
      </div>
    );
  }

  return (
    <div className="dashboard">
      {/* Header */}
      <header className="dashboard-header">
        <div className="logo-container">
          <img src="/logo.png" alt="App Logo" className="logo-icon" style={{ width: '48px', height: '48px', borderRadius: '12px', objectFit: 'contain', animation: 'pulse 3s infinite' }} />
          <div className="dashboard-title">
            <h1>Customer Analytics Dashboard</h1>
            <p>XGBoost Churn Forecast & Customer Segmentation Engine</p>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '12px', flexWrap: 'wrap' }}>
          <button 
            className="btn" 
            style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '6px 12px', borderRadius: '8px' }}
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            aria-label="Toggle dark and light theme"
          >
            {theme === 'dark' ? <Sun size={16} style={{ color: 'var(--color-med)' }} /> : <Moon size={16} style={{ color: 'var(--color-blue)' }} />}
            <span style={{ fontSize: '13px', fontWeight: 600 }}>{theme === 'dark' ? 'Light Mode' : 'Dark Mode'}</span>
          </button>
          <span className="badge" style={{ backgroundColor: 'var(--bg-subtle)', color: 'var(--text-primary)', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '6px 12px' }}>
            Model Status: Tuned
          </span>
          <span className="badge badge-high" style={{ display: 'flex', alignItems: 'center', gap: '4px', borderRadius: '8px', padding: '6px 12px' }}>
            <Sparkles size={12} /> Recall 81%
          </span>
        </div>
      </header>

      {/* Navigation Tab Bar */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', borderBottom: '1px solid var(--border-color)', paddingBottom: '12px', marginTop: '-8px', marginBottom: '8px', flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginRight: '16px' }}>
          <img src="/logo.png" alt="Logo" style={{ width: '22px', height: '22px', borderRadius: '6px', objectFit: 'contain' }} />
          <span style={{ fontSize: '13px', fontWeight: 800, fontFamily: 'var(--font-heading)', color: 'var(--color-blue)', letterSpacing: '0.5px' }}>CHURNFLOW</span>
        </div>
        <button 
          onClick={() => setActiveTab('insights')}
          style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '8px', 
            border: 'none', 
            cursor: 'pointer',
            backgroundColor: activeTab === 'insights' ? 'var(--color-blue-bg)' : 'transparent', 
            color: activeTab === 'insights' ? 'var(--color-blue)' : 'var(--text-secondary)', 
            padding: '8px 16px', 
            borderRadius: '8px', 
            fontWeight: 600,
            transition: 'all 0.2s'
          }}
        >
          <Users size={16} /> Database Insights
        </button>
        <button 
          onClick={() => setActiveTab('calculator')}
          style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '8px', 
            border: 'none', 
            cursor: 'pointer',
            backgroundColor: activeTab === 'calculator' ? 'var(--color-blue-bg)' : 'transparent', 
            color: activeTab === 'calculator' ? 'var(--color-blue)' : 'var(--text-secondary)', 
            padding: '8px 16px', 
            borderRadius: '8px', 
            fontWeight: 600,
            transition: 'all 0.2s'
          }}
        >
          <Calculator size={16} /> Real-time Calculator
        </button>
        <button 
          onClick={() => setActiveTab('monitor')}
          style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '8px', 
            border: 'none', 
            cursor: 'pointer',
            backgroundColor: activeTab === 'monitor' ? 'var(--color-blue-bg)' : 'transparent', 
            color: activeTab === 'monitor' ? 'var(--color-blue)' : 'var(--text-secondary)', 
            padding: '8px 16px', 
            borderRadius: '8px', 
            fontWeight: 600,
            transition: 'all 0.2s'
          }}
        >
          <Activity size={16} /> Model Health / Drift
        </button>
      </div>

      {/* Tab 1: Database Insights View */}
      {activeTab === 'insights' && (
        <>
          {/* KPI Section */}
          {summary && (
            <section className="kpi-grid">
              <MetricCard 
                title="Total Customers" 
                value={summary.total_customers.toLocaleString()} 
                subtext="Active in Database" 
                icon={<Users size={20} />} 
                iconColor="var(--color-blue)"
              />
              <MetricCard 
                title="Revenue At Risk" 
                value={`$${summary.revenue_at_risk.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} 
                subtext="Total value of High Risk customers" 
                icon={<DollarSign size={20} />} 
                iconColor="var(--color-high)"
              />
              <MetricCard 
                title="High Churn Risk" 
                value={summary.risk_tiers["High Risk"].toLocaleString()} 
                subtext={`${((summary.risk_tiers["High Risk"] / summary.total_customers) * 100).toFixed(1)}% of customer base`} 
                icon={<AlertTriangle size={20} />} 
                iconColor="var(--color-high)"
              />
              <MetricCard 
                title="Average Risk" 
                value={`${((summary.risk_tiers["High Risk"] + summary.risk_tiers["Medium Risk"] * 0.5) / summary.total_customers * 100).toFixed(1)}%`} 
                subtext="Calculated database risk score" 
                icon={<TrendingUp size={20} />} 
                iconColor="var(--color-low)"
              />
            </section>
          )}

          {/* Charts Section */}
          {summary && (
            <section className="charts-grid">
              <RiskDistributionChart 
                high={summary.risk_tiers["High Risk"]}
                med={summary.risk_tiers["Medium Risk"]}
                low={summary.risk_tiers["Low Risk"]}
                activeFilter={riskFilter}
                onFilterChange={handleRiskFilterChange}
              />
              <SegmentBarChart 
                segments={summary.segment_distribution}
                activeFilter={segmentFilter}
                onFilterChange={handleSegmentFilterChange}
              />
            </section>
          )}

          {/* Live Register Streaming Feed */}
          <section className="card" style={{ marginBottom: '24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '12px' }}>
              <div>
                <h3 style={{ fontFamily: 'var(--font-heading)', fontSize: '18px', fontWeight: 700, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Play size={18} style={{ color: 'var(--color-low)' }} /> Real-Time Customer Transaction Stream (WebSockets)
                </h3>
                <p style={{ color: 'var(--text-secondary)', fontSize: '12px', marginTop: '2px' }}>
                  Simulates live order flows from retail registers. Customers' RFM features and churn risk scores update dynamically.
                </p>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px' }}>
                  <span style={{ 
                    width: '8px', 
                    height: '8px', 
                    borderRadius: '50%', 
                    backgroundColor: wsStatus === 'connected' ? 'var(--color-low)' : wsStatus === 'connecting' ? 'var(--color-medium)' : 'var(--text-muted)',
                    boxShadow: wsStatus === 'connected' ? '0 0 8px var(--color-low)' : 'none',
                    display: 'inline-block'
                  }}></span>
                  <span style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>
                    {wsStatus === 'connected' ? 'Live Stream Active' : wsStatus === 'connecting' ? 'Connecting...' : 'Stream Paused'}
                  </span>
                </div>
                <button 
                  className="btn" 
                  onClick={toggleWebSocket}
                  style={{ 
                    cursor: 'pointer',
                    fontSize: '12px', 
                    padding: '8px 16px', 
                    borderRadius: '8px', 
                    fontWeight: 600,
                    backgroundColor: wsStatus === 'connected' ? 'var(--color-high-bg)' : 'var(--color-low-bg)',
                    borderColor: wsStatus === 'connected' ? 'var(--color-high-border)' : 'var(--color-low-border)',
                    color: wsStatus === 'connected' ? 'var(--color-high)' : 'var(--color-low)'
                  }}
                >
                  {wsStatus === 'connected' ? 'Pause Live Stream' : 'Connect Real-time Stream'}
                </button>
              </div>
            </div>

            {liveLogs.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '32px 0', color: 'var(--text-muted)', fontSize: '13px', border: '1px dashed var(--border-color)', borderRadius: '8px' }}>
                WebSocket feed idle. Click "Connect Real-time Stream" to simulate live cash register transactions.
              </div>
            ) : (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '12px', maxHeight: '180px', overflowY: 'auto', padding: '4px' }}>
                {liveLogs.map((log: any, index: number) => {
                  const isNewest = index === 0;
                  return (
                    <div 
                      key={log.id + '-' + index}
                      className="card animate-fade-in"
                      style={{ 
                        padding: '12px 14px', 
                        fontSize: '12px', 
                        margin: 0,
                        backgroundColor: isNewest ? 'var(--bg-subtle)' : 'transparent',
                        borderColor: isNewest ? 'var(--color-low-border)' : 'var(--border-color)',
                        borderLeft: `4px solid ${log.newMetrics.riskTier === 'High Risk' ? 'var(--color-high)' : log.newMetrics.riskTier === 'Medium Risk' ? 'var(--color-medium)' : 'var(--color-low)'}`
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontWeight: 600, marginBottom: '4px' }}>
                        <span>Customer #{log.id}</span>
                        <span style={{ color: 'var(--color-low)' }}>+${log.invoiceValue.toFixed(2)}</span>
                      </div>
                      <div style={{ color: 'var(--text-secondary)', fontSize: '11px', lineHeight: '1.4' }}>
                        <div>Ordered <strong>{log.quantity}</strong> items (Freq: {log.newMetrics.frequency})</div>
                        <div style={{ marginTop: '4px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span>Churn Risk: <strong>{(log.newMetrics.churnProb * 100).toFixed(1)}%</strong></span>
                          <span style={{ 
                            padding: '2px 6px', 
                            fontSize: '9px', 
                            fontWeight: 700, 
                            borderRadius: '4px',
                            color: log.newMetrics.riskTier === 'High Risk' ? 'var(--color-high)' : log.newMetrics.riskTier === 'Medium Risk' ? 'var(--color-medium)' : 'var(--color-low)',
                            backgroundColor: log.newMetrics.riskTier === 'High Risk' ? 'var(--color-high-bg)' : log.newMetrics.riskTier === 'Medium Risk' ? 'var(--color-medium-bg)' : 'var(--color-low-bg)'
                          }}>
                            {log.newMetrics.riskTier}
                          </span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </section>

          {/* Search and Filters Bar */}
          <section className="search-filter-section">
            <div className="search-box" ref={searchRef}>
              <Search className="search-icon" size={18} />
              <input 
                type="text" 
                className="search-input" 
                placeholder="Search by Customer ID (e.g. 12918, 15760)..."
                value={searchQuery}
                onChange={(e) => {
                  setSearchQuery(e.target.value);
                  setShowSearchResults(true);
                  setCurrentPage(1);
                }}
                onFocus={() => setShowSearchResults(true)}
              />
              {showSearchResults && searchSuggestions.length > 0 && (
                <div className="search-results">
                  {searchSuggestions.map((customer) => (
                    <div 
                      key={customer.id} 
                      className="search-result-item"
                      onClick={() => handleSelectSuggestion(customer)}
                    >
                      <span className="search-result-id">{customer.id}</span>
                      <span className="search-result-segment">{customer.segment} ({customer.riskTier})</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="filter-group">
              {/* Segment Filter */}
              <select 
                className="filter-select"
                value={segmentFilter || ""}
                onChange={(e) => handleSegmentFilterChange(e.target.value || null)}
              >
                <option value="">All Segments</option>
                {summary && Object.keys(summary.segment_distribution).map((seg) => (
                  <option key={seg} value={seg}>{seg}</option>
                ))}
              </select>

              {/* Risk Filter */}
              <select 
                className="filter-select"
                value={riskFilter || ""}
                onChange={(e) => handleRiskFilterChange(e.target.value || null)}
              >
                <option value="">All Risk Tiers</option>
                <option value="High Risk">High Risk</option>
                <option value="Medium Risk">Medium Risk</option>
                <option value="Low Risk">Low Risk</option>
              </select>
            </div>
          </section>

          {/* Main Workspace (Table & Detail Lookup) */}
          <section className="workspace-grid">
            <CustomerTable 
              customers={sortedCustomers}
              selectedCustomerId={selectedCustomer ? selectedCustomer.id : null}
              onSelectCustomer={(c) => setSelectedCustomer(c)}
              currentPage={currentPage}
              setCurrentPage={setCurrentPage}
              pageSize={pageSize}
              sortField={sortField}
              sortDirection={sortDirection}
              onSort={handleSort}
            />
            <CustomerDetails 
              customer={selectedCustomer}
            />
          </section>
        </>
      )}

      {/* Tab 2: Real-time Calculator View */}
      {activeTab === 'calculator' && (
        <section className="workspace-grid" style={{ gridTemplateColumns: '1fr 1.2fr' }}>
          {/* Form Card */}
          <div className="card">
            <h3 style={{ fontFamily: 'var(--font-heading)', fontSize: '20px', fontWeight: 700, marginBottom: '20px', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '10px' }}>
              <Calculator style={{ color: 'var(--color-blue)' }} /> Customer Profile Metrics
            </h3>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              {/* Recency */}
              <div>
                <label style={{ display: 'block', fontSize: '11px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  Recency (Days since last purchase)
                </label>
                <input 
                  type="number" 
                  className="search-input" 
                  style={{ paddingLeft: '16px' }}
                  value={calcRecency} 
                  onChange={(e) => setCalcRecency(Math.max(0, parseInt(e.target.value) || 0))} 
                />
              </div>

              {/* Frequency */}
              <div>
                <label style={{ display: 'block', fontSize: '11px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  Frequency (Total unique invoices)
                </label>
                <input 
                  type="number" 
                  className="search-input" 
                  style={{ paddingLeft: '16px' }}
                  value={calcFrequency} 
                  onChange={(e) => setCalcFrequency(Math.max(1, parseInt(e.target.value) || 1))} 
                />
              </div>

              {/* Monetary */}
              <div>
                <label style={{ display: 'block', fontSize: '11px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  Monetary Value (Average spend per invoice, $)
                </label>
                <input 
                  type="number" 
                  className="search-input" 
                  style={{ paddingLeft: '16px' }}
                  value={calcMonetary} 
                  onChange={(e) => setCalcMonetary(Math.max(0, parseFloat(e.target.value) || 0))} 
                />
              </div>

              {/* Basket Size */}
              <div>
                <label style={{ display: 'block', fontSize: '11px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  Average Basket Size (Avg items per order)
                </label>
                <input 
                  type="number" 
                  className="search-input" 
                  style={{ paddingLeft: '16px' }}
                  value={calcBasketSize} 
                  onChange={(e) => setCalcBasketSize(Math.max(0, parseFloat(e.target.value) || 0))} 
                />
              </div>

              {/* Advanced toggle group / fields */}
              <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '16px', marginTop: '4px' }}>
                <h4 style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '12px' }}>Advanced Feature Overrides</h4>
                
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '6px', textTransform: 'uppercase' }}>
                      Avg Days Between
                    </label>
                    <input 
                      type="text" 
                      className="search-input" 
                      style={{ paddingLeft: '12px' }}
                      value={calcAvgDaysBetween} 
                      placeholder="e.g. 30 (Optional)"
                      onChange={(e) => setCalcAvgDaysBetween(e.target.value)} 
                    />
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '6px', textTransform: 'uppercase' }}>
                      Recent Orders Ratio
                    </label>
                    <input 
                      type="text" 
                      className="search-input" 
                      style={{ paddingLeft: '12px' }}
                      value={calcRecentOrdersRatio} 
                      placeholder="e.g. 1.0 (Optional)"
                      onChange={(e) => setCalcRecentOrdersRatio(e.target.value)} 
                    />
                  </div>
                </div>

                <div style={{ marginTop: '12px' }}>
                  <label style={{ display: 'block', fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '6px', textTransform: 'uppercase' }}>
                    Customer Country (UK vs International)
                  </label>
                  <select 
                    className="filter-select"
                    style={{ width: '100%', padding: '12px' }}
                    value={calcIsUk}
                    onChange={(e) => setCalcIsUk(Number(e.target.value))}
                  >
                    <option value={1}>United Kingdom (UK)</option>
                    <option value={0}>International (Non-UK)</option>
                  </select>
                </div>
              </div>

              <button 
                className="btn"
                style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', padding: '14px', borderRadius: '10px', backgroundColor: 'var(--color-blue-bg)', color: 'var(--color-blue)', border: '1px solid var(--color-blue-border)', fontWeight: 700, fontSize: '14px', marginTop: '8px', cursor: 'pointer' }}
                onClick={runLiveForecast}
                disabled={calcLoading}
              >
                {calcLoading ? (
                  <>
                    <RefreshCw size={18} style={{ animation: 'spin 1.5s linear infinite' }} /> Calculating...
                  </>
                ) : (
                  <>
                    <Play size={16} fill="var(--color-blue)" /> Calculate Churn Risk
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Results Card */}
          <div className="card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', minHeight: '400px' }}>
            {calcLoading ? (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px' }}>
                <RefreshCw className="logo-icon" size={48} style={{ animation: 'spin 2s linear infinite' }} />
                <p style={{ color: 'var(--text-secondary)' }}>Querying XGBoost prediction engine...</p>
              </div>
            ) : calcError ? (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px', textAlign: 'center', padding: '20px' }}>
                <ShieldAlert size={48} style={{ color: 'var(--color-high)' }} />
                <h4 style={{ fontFamily: 'var(--font-heading)', fontSize: '18px', fontWeight: 600 }}>Connection Error</h4>
                <p style={{ color: 'var(--text-secondary)', fontSize: '13px', maxWidth: '300px' }}>{calcError}</p>
                <button className="btn" onClick={runLiveForecast} style={{ fontSize: '12px', padding: '8px 16px', marginTop: '8px' }}>
                  Retry Inference
                </button>
              </div>
            ) : calcResult ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                <div>
                  <h3 style={{ fontFamily: 'var(--font-heading)', fontSize: '20px', fontWeight: 700, marginBottom: '4px' }}>Inference Prediction Result</h3>
                  <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>Calculated dynamically by the tuned XGBoost model</p>
                </div>

                {/* Score gauge */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)' }}>CHURN PROBABILITY</span>
                    <span style={{ fontFamily: 'var(--font-heading)', fontSize: '24px', fontWeight: 800, color: calcResult.risk_tier === 'High Risk' ? 'var(--color-high)' : calcResult.risk_tier === 'Medium Risk' ? 'var(--color-med)' : 'var(--color-low)' }}>
                      {(calcResult.churn_probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  
                  {/* Gauge bar */}
                  <div style={{ height: '16px', backgroundColor: 'var(--border-color)', borderRadius: '8px', overflow: 'hidden' }}>
                    <div style={{ 
                      height: '100%', 
                      width: `${calcResult.churn_probability * 100}%`, 
                      backgroundColor: calcResult.risk_tier === 'High Risk' ? 'var(--color-high)' : calcResult.risk_tier === 'Medium Risk' ? 'var(--color-med)' : 'var(--color-low)',
                      borderRadius: '8px',
                      transition: 'width 0.8s ease-out'
                    }} />
                  </div>
                  
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-muted)' }}>
                    <span>0% (LOYAL)</span>
                    <span>100% (CHURNED)</span>
                  </div>
                </div>

                {/* Risk Tier Badge */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)' }}>RISK CLASSIFICATION:</span>
                  <span className={`badge ${calcResult.risk_tier === 'High Risk' ? 'badge-high' : calcResult.risk_tier === 'Medium Risk' ? 'badge-med' : 'badge-low'}`} style={{ padding: '6px 12px', fontSize: '12px', borderRadius: '8px' }}>
                    {calcResult.risk_tier}
                  </span>
                </div>

                {calcResult.shap_values && (
                  <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '16px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '4px' }}>
                      Model Decision Drivers (TreeSHAP Impact)
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      {Object.entries(calcResult.shap_values)
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

                {/* Recommendation box */}
                <div className={`recommendation-box ${calcResult.risk_tier === 'High Risk' ? 'high-risk' : calcResult.risk_tier === 'Medium Risk' ? 'med-risk' : ''}`} style={{ borderRadius: '12px' }}>
                  <AlertTriangle className="recommendation-icon" size={20} style={{ 
                    color: calcResult.risk_tier === 'High Risk' ? 'var(--color-high)' : calcResult.risk_tier === 'Medium Risk' ? 'var(--color-med)' : 'var(--color-blue)',
                    marginTop: '2px'
                  }} />
                  <div className="recommendation-content">
                    <h4 style={{ color: 'var(--text-primary)', fontSize: '14px', fontWeight: 700 }}>Retainment Strategy recommendation</h4>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '13px', lineHeight: '1.4', marginTop: '2px' }}>{calcResult.recommendation}</p>
                  </div>
                </div>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '16px', color: 'var(--text-muted)' }}>
                <Calculator size={64} style={{ color: 'var(--border-color)', strokeWidth: 1 }} />
                <div style={{ textAlign: 'center' }}>
                  <h4 style={{ color: 'var(--text-secondary)', fontSize: '15px', fontWeight: 600, marginBottom: '4px' }}>Awaiting Calculation</h4>
                  <p style={{ fontSize: '13px', maxWidth: '280px' }}>Enter parameters on the left and click 'Calculate Churn Risk' to run active inference forecast.</p>
                </div>
              </div>
            )}
          </div>
        </section>
      )}

      {/* Tab 3: Model Health / Drift View */}
      {activeTab === 'monitor' && (
        <section style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {/* Header Description */}
          <div className="card">
            <h3 style={{ fontFamily: 'var(--font-heading)', fontSize: '20px', fontWeight: 700, marginBottom: '8px', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '10px' }}>
              <Activity style={{ color: 'var(--color-blue)' }} /> Real-time Covariate Drift Monitoring (Kolmogorov-Smirnov Test)
            </h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '13px', lineHeight: '1.5', maxWidth: '850px' }}>
              Detects if production data distributions deviate statistically from baseline training distributions.
              A significance level of <strong>&alpha; = 0.05</strong> is utilized. If a feature's Kolmogorov-Smirnov 2-sample p-value drops below 0.05, it indicates covariate shift, warning that model accuracy may degrade due to distribution mismatch.
            </p>
          </div>

          {driftLoading ? (
            <div className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '300px', gap: '16px' }}>
              <RefreshCw className="logo-icon" size={48} style={{ animation: 'spin 2s linear infinite' }} />
              <p style={{ color: 'var(--text-secondary)' }}>Calculating Kolmogorov-Smirnov statistics across historical and production samples...</p>
            </div>
          ) : driftError ? (
            <div className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '300px', gap: '16px', textAlign: 'center' }}>
              <ShieldAlert size={48} style={{ color: 'var(--color-high)' }} />
              <h4 style={{ fontSize: '18px', fontWeight: 600 }}>Connection Error</h4>
              <p style={{ color: 'var(--text-secondary)', fontSize: '13px', maxWidth: '300px' }}>{driftError}</p>
              <button className="btn" onClick={checkModelDrift} style={{ fontSize: '12px', padding: '8px 16px', marginTop: '8px', cursor: 'pointer' }}>
                Retry Analysis
              </button>
            </div>
          ) : driftData ? (
            <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '24px' }}>
              {/* Drift Status Card */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                {/* Status indicator block */}
                <div className={`card`} style={{ 
                  borderLeft: `6px solid ${driftData.drift_status === "Insufficient Data" ? 'var(--text-muted)' : driftData.drift_detected ? 'var(--color-high)' : 'var(--color-low)'}`,
                  padding: '24px'
                }}>
                  <div style={{ display: 'flex', alignItems: 'flex-start', gap: '16px' }}>
                    {driftData.drift_status === "Insufficient Data" ? (
                      <ShieldAlert size={36} style={{ color: 'var(--text-muted)' }} />
                    ) : driftData.drift_detected ? (
                      <AlertTriangle size={36} style={{ color: 'var(--color-high)' }} />
                    ) : (
                      <CheckCircle size={36} style={{ color: 'var(--color-low)' }} />
                    )}
                    <div>
                      <h4 style={{ fontFamily: 'var(--font-heading)', fontSize: '18px', fontWeight: 700, color: 'var(--text-primary)' }}>
                        {driftData.drift_status === "Insufficient Data" ? "Awaiting Data" : driftData.drift_detected ? "Covariate Shift Detected!" : "Model Status: Stable"}
                      </h4>
                      <p style={{ color: 'var(--text-secondary)', fontSize: '13px', lineHeight: '1.4', marginTop: '6px' }}>
                        {driftData.message}
                      </p>
                      
                      {driftData.sample_sizes && (
                        <div style={{ display: 'flex', gap: '24px', marginTop: '16px', fontSize: '12px', color: 'var(--text-muted)' }}>
                          <span>Baseline dataset size: <strong>{driftData.sample_sizes.baseline}</strong> records</span>
                          <span>Production logs: <strong>{driftData.sample_sizes.production}</strong> predictions</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Simulation block */}
                <div className="card">
                  <h4 style={{ fontFamily: 'var(--font-heading)', fontSize: '16px', fontWeight: 700, marginBottom: '10px' }}>MLOps Sandbox: Simulate Feature Drift</h4>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '13px', lineHeight: '1.4', marginBottom: '16px' }}>
                    Inject 15 mock customers with heavily shifted transactional metrics (e.g. extremely high spend and basket size) directly into production logs. This will immediately trigger statistical drift alerts!
                  </p>
                  <div style={{ display: 'flex', gap: '12px' }}>
                    <button 
                      className="btn"
                      style={{ cursor: 'pointer', padding: '10px 20px', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--color-high)', borderColor: 'var(--color-high-border)', backgroundColor: 'var(--color-high-bg)', fontWeight: 600, fontSize: '13px', borderRadius: '8px' }}
                      onClick={injectDriftData}
                      disabled={driftLoading}
                    >
                      <Activity size={16} /> Inject Out-of-Distribution Data
                    </button>
                    <button 
                      className="btn"
                      style={{ cursor: 'pointer', padding: '10px 20px', display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 600, fontSize: '13px', borderRadius: '8px' }}
                      onClick={checkModelDrift}
                      disabled={driftLoading}
                    >
                      <RefreshCw size={16} /> Recalculate Metrics
                    </button>
                  </div>
                </div>

                {/* Shadow Deployment Card */}
                <div className="card">
                  <h4 style={{ fontFamily: 'var(--font-heading)', fontSize: '16px', fontWeight: 700, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Activity size={18} style={{ color: 'var(--color-blue)' }} /> Shadow Deployment: Champion vs Challenger
                  </h4>
                  
                  {shadowLoading ? (
                    <div style={{ display: 'flex', justifyContent: 'center', padding: '24px' }}>
                      <RefreshCw size={24} style={{ animation: 'spin 2s linear infinite' }} />
                    </div>
                  ) : shadowError ? (
                    <div style={{ color: 'var(--color-high)', fontSize: '12px', padding: '8px 0' }}>{shadowError}</div>
                  ) : shadowData && shadowData.total_predictions > 0 ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                        <div style={{ padding: '12px', border: '1px solid var(--border-color)', borderRadius: '8px', backgroundColor: 'var(--bg-subtle)' }}>
                          <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Active Champion</div>
                          <div style={{ fontWeight: 700, fontSize: '14px', marginTop: '2px' }}>XGBoost Classifier</div>
                          <div style={{ fontSize: '12px', color: 'var(--color-low)', fontWeight: 600, marginTop: '2px' }}>Avg Prob: {(shadowData.champion_mean * 100).toFixed(1)}%</div>
                        </div>
                        <div style={{ padding: '12px', border: '1px solid var(--border-color)', borderRadius: '8px', backgroundColor: 'var(--bg-subtle)' }}>
                          <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Shadow Challenger</div>
                          <div style={{ fontWeight: 700, fontSize: '14px', marginTop: '2px' }}>Random Forest</div>
                          <div style={{ fontSize: '12px', color: 'var(--color-low)', fontWeight: 600, marginTop: '2px' }}>Avg Prob: {(shadowData.challenger_mean * 100).toFixed(1)}%</div>
                        </div>
                      </div>

                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', fontSize: '13px', borderTop: '1px solid var(--border-color)', paddingTop: '12px' }}>
                        <div>Prediction Agreement: <strong>{(shadowData.agreement_rate * 100).toFixed(1)}%</strong></div>
                        <div>Mean Abs Deviation (MAD): <strong>{shadowData.mean_absolute_deviation.toFixed(4)}</strong></div>
                      </div>

                      {/* Recent shadow logs list */}
                      <div>
                        <h5 style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '8px', textTransform: 'uppercase' }}>Recent Execution Comparison Logs</h5>
                        <div style={{ overflowX: 'auto' }}>
                          <table style={{ width: '100%', fontSize: '11px', borderCollapse: 'collapse', textAlign: 'left' }}>
                            <thead>
                              <tr style={{ borderBottom: '1px solid var(--border-color)', color: 'var(--text-muted)' }}>
                                <th style={{ padding: '6px 4px' }}>Time</th>
                                <th style={{ padding: '6px 4px' }}>RFM Profile</th>
                                <th style={{ padding: '6px 4px', textAlign: 'right' }}>Champion</th>
                                <th style={{ padding: '6px 4px', textAlign: 'right' }}>Challenger</th>
                              </tr>
                            </thead>
                            <tbody>
                              {shadowData.recent_logs.map((log: any) => (
                                <tr key={log.id} style={{ borderBottom: '1px solid var(--border-color)' }}>
                                  <td style={{ padding: '8px 4px', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>{log.timestamp.split(' ')[1]}</td>
                                  <td style={{ padding: '8px 4px' }}>R{log.recency}/F{log.frequency}/M${Math.round(log.monetary)}</td>
                                  <td style={{ padding: '8px 4px', textAlign: 'right', fontWeight: 600 }}>{(log.champion_prob * 100).toFixed(0)}%</td>
                                  <td style={{ padding: '8px 4px', textAlign: 'right', fontWeight: 600 }}>{(log.challenger_prob * 100).toFixed(0)}%</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div style={{ textAlign: 'center', padding: '24px 0', color: 'var(--text-muted)', fontSize: '13px' }}>
                      <Users size={32} style={{ margin: '0 auto 8px', color: 'var(--border-color)' }} />
                      Awaiting live forecasts. Execute a prediction in the <strong>Real-time Calculator</strong> tab to see shadow model comparison logs.
                    </div>
                  )}
                </div>
              </div>

              {/* Feature details table */}
              <div className="card">
                <h4 style={{ fontFamily: 'var(--font-heading)', fontSize: '16px', fontWeight: 700, marginBottom: '16px' }}>Statistical Feature Breakdown</h4>
                
                {driftData.features ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    {Object.entries(driftData.features).map(([feat, details]: any) => {
                      const isDrifted = details.drift_status === "Drifted";
                      return (
                        <div key={feat} style={{ padding: '14px', border: '1px solid var(--border-color)', borderRadius: '10px', backgroundColor: 'var(--bg-subtle)' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                            <span style={{ fontWeight: 700, fontSize: '14px', color: 'var(--text-primary)' }}>{feat}</span>
                            <span className={`badge ${isDrifted ? 'badge-high' : 'badge-low'}`} style={{ fontSize: '11px', padding: '4px 8px', borderRadius: '6px' }}>
                              {details.drift_status}
                            </span>
                          </div>
                          
                          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '12px', color: 'var(--text-secondary)' }}>
                            <div>Baseline Mean: <strong>{feat === "Monetary" ? "$" : ""}{details.baseline_mean.toLocaleString()}</strong></div>
                            <div>Production Mean: <strong>{feat === "Monetary" ? "$" : ""}{details.production_mean.toLocaleString()}</strong></div>
                          </div>
                          
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-muted)', marginTop: '8px', borderTop: '1px solid var(--border-color)', paddingTop: '6px' }}>
                            <span>KS Test p-value</span>
                            <span style={{ fontWeight: 600, color: isDrifted ? 'var(--color-high)' : 'var(--text-muted)' }}>{details.p_value.toFixed(5)}</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '24px 0' }}>
                    Awaiting production samples to display individual feature statistics.
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '300px', gap: '16px' }}>
              <RefreshCw size={24} style={{ animation: 'spin 2s linear infinite' }} />
              <button className="btn" onClick={checkModelDrift} style={{ cursor: 'pointer' }}>Initialize Health Audit</button>
            </div>
          )}
        </section>
      )}
    </div>
  );
}

export default App;
