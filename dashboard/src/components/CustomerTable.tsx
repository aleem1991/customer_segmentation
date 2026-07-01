import React from 'react';
import { ChevronLeft, ChevronRight, ArrowUpDown } from 'lucide-react';

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

interface CustomerTableProps {
  customers: CustomerRecord[];
  selectedCustomerId: string | null;
  onSelectCustomer: (customer: CustomerRecord) => void;
  currentPage: number;
  setCurrentPage: (page: number) => void;
  pageSize: number;
  sortField: string;
  sortDirection: 'asc' | 'desc';
  onSort: (field: any) => void;
}

export const CustomerTable: React.FC<CustomerTableProps> = ({
  customers,
  selectedCustomerId,
  onSelectCustomer,
  currentPage,
  setCurrentPage,
  pageSize,
  sortField,
  sortDirection,
  onSort
}) => {
  const totalRecords = customers.length;
  const totalPages = Math.ceil(totalRecords / pageSize) || 1;
  
  // Slice data for current page
  const startIndex = (currentPage - 1) * pageSize;
  const endIndex = Math.min(startIndex + pageSize, totalRecords);
  const paginatedData = customers.slice(startIndex, endIndex);

  // Helper to render sort arrow
  const renderSortIndicator = (field: string) => {
    if (sortField === field) {
      return (
        <span style={{ marginLeft: '4px', display: 'inline-block', fontSize: '10px', color: 'var(--color-blue)' }}>
          {sortDirection === 'asc' ? '▲' : '▼'}
        </span>
      );
    }
    return <ArrowUpDown size={10} style={{ marginLeft: '4px', opacity: 0.5 }} />;
  };

  const getRiskBadge = (tier: string) => {
    if (tier === 'High Risk') return <span className="badge badge-high">High</span>;
    if (tier === 'Medium Risk') return <span className="badge badge-med">Medium</span>;
    return <span className="badge badge-low">Low</span>;
  };

  return (
    <div className="card table-card">
      <div className="table-header-row">
        <div className="chart-title">Customer List ({totalRecords})</div>
        <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
          Showing {totalRecords > 0 ? startIndex + 1 : 0}-{endIndex} of {totalRecords}
        </div>
      </div>

      <div className="table-container">
        <table>
          <thead>
            <tr>
              <th onClick={() => onSort('id')}>Customer ID {renderSortIndicator('id')}</th>
              <th onClick={() => onSort('segment')}>Segment {renderSortIndicator('segment')}</th>
              <th onClick={() => onSort('recency')} style={{ textAlign: 'right' }}>Recency {renderSortIndicator('recency')}</th>
              <th onClick={() => onSort('frequency')} style={{ textAlign: 'right' }}>Freq {renderSortIndicator('frequency')}</th>
              <th onClick={() => onSort('monetary')} style={{ textAlign: 'right' }}>Avg Spend {renderSortIndicator('monetary')}</th>
              <th onClick={() => onSort('churnProb')} style={{ textAlign: 'right' }}>Churn Risk {renderSortIndicator('churnProb')}</th>
              <th style={{ textAlign: 'center' }}>Status</th>
            </tr>
          </thead>
          <tbody>
            {paginatedData.length === 0 ? (
              <tr>
                <td colSpan={7} style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>
                  No customer records match the active filters.
                </td>
              </tr>
            ) : (
              paginatedData.map((customer) => (
                <tr 
                  key={customer.id} 
                  className={`table-row ${selectedCustomerId === customer.id ? 'selected' : ''}`}
                  onClick={() => onSelectCustomer(customer)}
                >
                  <td style={{ fontWeight: '600' }}>{customer.id}</td>
                  <td>{customer.segment}</td>
                  <td style={{ textAlign: 'right' }}>{customer.recency}d</td>
                  <td style={{ textAlign: 'right' }}>{customer.frequency}</td>
                  <td style={{ textAlign: 'right', fontWeight: '500' }}>
                    ${customer.monetary.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </td>
                  <td style={{ textAlign: 'right', fontWeight: '600', color: customer.churnProb >= 0.7 ? 'var(--color-high)' : customer.churnProb >= 0.3 ? 'var(--color-med)' : 'var(--color-low)' }}>
                    {(customer.churnProb * 100).toFixed(1)}%
                  </td>
                  <td style={{ textAlign: 'center' }}>
                    {getRiskBadge(customer.riskTier)}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      <div className="pagination">
        <div>
          Page <strong>{currentPage}</strong> of <strong>{totalPages}</strong>
        </div>
        <div className="pagination-buttons">
          <button 
            className="btn" 
            onClick={() => setCurrentPage(currentPage - 1)}
            disabled={currentPage <= 1}
          >
            <ChevronLeft size={16} style={{ display: 'block' }} />
          </button>
          <button 
            className="btn" 
            onClick={() => setCurrentPage(currentPage + 1)}
            disabled={currentPage >= totalPages}
          >
            <ChevronRight size={16} style={{ display: 'block' }} />
          </button>
        </div>
      </div>
    </div>
  );
};
