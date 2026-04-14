import React, { createContext, useContext, useState, useCallback } from 'react';

const AppContext = createContext();

export const useAppState = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useAppState must be used within AppProvider');
  }
  return context;
};

export const AppProvider = ({ children }) => {
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem('theme');
    return saved || 'light';
  });

  const [autoRefresh, setAutoRefresh] = useState(() => {
    const saved = localStorage.getItem('autoRefresh');
    return saved ? JSON.parse(saved) : true;
  });

  const [refreshInterval, setRefreshInterval] = useState(() => {
    const saved = localStorage.getItem('refreshInterval');
    return saved ? parseInt(saved) : 30000;
  });

  const [soundEnabled, setSoundEnabled] = useState(() => {
    const saved = localStorage.getItem('soundEnabled');
    return saved ? JSON.parse(saved) : true;
  });

  const [notificationPreferences, setNotificationPreferences] = useState(() => {
    const saved = localStorage.getItem('notificationPreferences');
    return saved ? JSON.parse(saved) : {
      email: true,
      push: true,
      highPriorityOnly: false,
    };
  });

  const updateTheme = useCallback((newTheme) => {
    setTheme(newTheme);
    localStorage.setItem('theme', newTheme);
  }, []);

  const updateAutoRefresh = useCallback((value) => {
    setAutoRefresh(value);
    localStorage.setItem('autoRefresh', JSON.stringify(value));
  }, []);

  const updateRefreshInterval = useCallback((value) => {
    setRefreshInterval(value);
    localStorage.setItem('refreshInterval', value.toString());
  }, []);

  const updateSoundEnabled = useCallback((value) => {
    setSoundEnabled(value);
    localStorage.setItem('soundEnabled', JSON.stringify(value));
  }, []);

  const updateNotificationPreferences = useCallback((prefs) => {
    setNotificationPreferences(prev => ({
      ...prev,
      ...prefs,
    }));
    localStorage.setItem('notificationPreferences', JSON.stringify({
      ...notificationPreferences,
      ...prefs,
    }));
  }, [notificationPreferences]);

  const value = {
    theme,
    updateTheme,
    autoRefresh,
    updateAutoRefresh,
    refreshInterval,
    updateRefreshInterval,
    soundEnabled,
    updateSoundEnabled,
    notificationPreferences,
    updateNotificationPreferences,
  };

  return (
    <AppContext.Provider value={value}>
      {children}
    </AppContext.Provider>
  );
};
