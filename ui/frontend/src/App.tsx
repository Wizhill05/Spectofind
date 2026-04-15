import { useState } from 'react';
import Dashboard from './components/Dashboard';
import Studio from './components/Studio';

type Tab = 'dashboard' | 'studio';

export default function App() {
  const [tab, setTab] = useState<Tab>('studio');

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header__logo">
          <div className="header__icon">
            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 4L12 20M17 8L17 16M7 8L7 16M22 10L22 14M2 10L2 14" stroke="currentColor" strokeWidth="2" strokeLinecap="square"/>
            </svg>
          </div>
          <div>
            <div className="header__title">Spectofind</div>
            <div className="header__subtitle">Core Classification Engine</div>
          </div>
        </div>

        <div className="tab-bar">
          <button
            id="tab-studio"
            className={`tab-btn ${tab === 'studio' ? 'tab-btn--active' : ''}`}
            onClick={() => setTab('studio')}
          >
            [ Studio ]
          </button>
          <button
            id="tab-dashboard"
            className={`tab-btn ${tab === 'dashboard' ? 'tab-btn--active' : ''}`}
            onClick={() => setTab('dashboard')}
          >
            [ Dashboard ]
          </button>
        </div>
      </header>

      {/* Tab Content */}
      {tab === 'dashboard' ? <Dashboard /> : <Studio />}
    </div>
  );
}
