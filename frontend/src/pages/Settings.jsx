import React, { useState, useEffect } from 'react';
import { Settings as SettingsIcon, Save, RotateCcw, Info } from 'lucide-react';
import { useAppState } from '../context/AppContext';
import { useToast } from '../context/ToastContext';
import { config } from '../config';

const Settings = () => {
  const appState = useAppState();
  const toast = useToast();
  const [saving, setSaving] = useState(false);
  
  const [settings, setSettings] = useState({
    notifications: {
      email: appState.notificationPreferences.email,
      push: appState.notificationPreferences.push,
      sound: appState.soundEnabled,
      highPriorityOnly: appState.notificationPreferences.highPriorityOnly,
    },
    dashboard: {
      refreshInterval: appState.refreshInterval / 1000,
      autoRefresh: appState.autoRefresh,
    },
    system: {
      theme: appState.theme,
      appVersion: config.app.version,
      apiUrl: config.api.baseURL,
      environment: config.app.environment,
    },
  });

  const handleSettingChange = (category, setting, value) => {
    setSettings(prev => ({
      ...prev,
      [category]: {
        ...prev[category],
        [setting]: value,
      },
    }));
  };

  const handleSave = async () => {
    try {
      setSaving(true);
      
      // Update context/local storage
      appState.updateNotificationPreferences({
        email: settings.notifications.email,
        push: settings.notifications.push,
        highPriorityOnly: settings.notifications.highPriorityOnly,
      });
      appState.updateSoundEnabled(settings.notifications.sound);
      appState.updateAutoRefresh(settings.dashboard.autoRefresh);
      appState.updateRefreshInterval(settings.dashboard.refreshInterval * 1000);
      appState.updateTheme(settings.system.theme);

      toast.success('Settings saved successfully');
    } catch (error) {
      toast.error('Failed to save settings');
      console.error(error);
    } finally {
      setSaving(false);
    }
  };

  const handleReset = () => {
    setSettings({
      notifications: {
        email: true,
        push: true,
        sound: true,
        highPriorityOnly: false,
      },
      dashboard: {
        refreshInterval: 30,
        autoRefresh: true,
      },
      system: {
        theme: 'light',
        appVersion: config.app.version,
        apiUrl: config.api.baseURL,
        environment: config.app.environment,
      },
    });
    toast.info('Settings reset to defaults');
  };

  const ToggleSwitch = ({ checked, onChange, disabled = false }) => (
    <button
      onClick={() => !disabled && onChange(!checked)}
      disabled={disabled}
      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
        checked ? 'bg-primary-600' : 'bg-gray-200'
      } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
    >
      <span
        className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
          checked ? 'translate-x-6' : 'translate-x-1'
        }`}
      />
    </button>
  );

  return (
    <div className="p-6">
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Settings</h1>
            <p className="text-gray-600 mt-2">Configure your construction AI system</p>
          </div>
          <div className="flex space-x-3">
            <button
              onClick={handleReset}
              className="btn btn-secondary flex items-center"
            >
              <RotateCcw className="h-4 w-4 mr-2" />
              Reset
            </button>
            <button
              onClick={handleSave}
              disabled={saving}
              className="btn btn-primary flex items-center"
            >
              {saving ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent mr-2" />
                  Saving...
                </>
              ) : (
                <>
                  <Save className="h-4 w-4 mr-2" />
                  Save Settings
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Notification Settings */}
        <div className="card">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">🔔 Notifications</h2>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <label className="text-sm font-medium text-gray-900">Email Notifications</label>
                <p className="text-sm text-gray-500">Receive alerts via email</p>
              </div>
              <ToggleSwitch
                checked={settings.notifications.email}
                onChange={(value) => handleSettingChange('notifications', 'email', value)}
              />
            </div>

            <div className="flex items-center justify-between">
              <div>
                <label className="text-sm font-medium text-gray-900">Push Notifications</label>
                <p className="text-sm text-gray-500">Receive browser push notifications</p>
              </div>
              <ToggleSwitch
                checked={settings.notifications.push}
                onChange={(value) => handleSettingChange('notifications', 'push', value)}
              />
            </div>

            <div className="flex items-center justify-between">
              <div>
                <label className="text-sm font-medium text-gray-900">Sound Alerts</label>
                <p className="text-sm text-gray-500">Play sound for high-priority alerts</p>
              </div>
              <ToggleSwitch
                checked={settings.notifications.sound}
                onChange={(value) => handleSettingChange('notifications', 'sound', value)}
              />
            </div>

            <div className="flex items-center justify-between pt-4 border-t border-gray-200">
              <div>
                <label className="text-sm font-medium text-gray-900">High Priority Only</label>
                <p className="text-sm text-gray-500">Only notify for high-priority alerts</p>
              </div>
              <ToggleSwitch
                checked={settings.notifications.highPriorityOnly}
                onChange={(value) => handleSettingChange('notifications', 'highPriorityOnly', value)}
              />
            </div>
          </div>
        </div>

        {/* Dashboard Settings */}
        <div className="card">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">📊 Dashboard</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-900 mb-2">Refresh Interval</label>
              <select
                value={settings.dashboard.refreshInterval}
                onChange={(e) => handleSettingChange('dashboard', 'refreshInterval', parseInt(e.target.value))}
                className="block w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              >
                <option value={10}>10 seconds</option>
                <option value={15}>15 seconds</option>
                <option value={30}>30 seconds</option>
                <option value={60}>1 minute</option>
                <option value={300}>5 minutes</option>
              </select>
              <p className="text-sm text-gray-500 mt-1">How often to refresh dashboard data</p>
            </div>

            <div className="flex items-center justify-between pt-4 border-t border-gray-200">
              <div>
                <label className="text-sm font-medium text-gray-900">Auto Refresh</label>
                <p className="text-sm text-gray-500">Automatically refresh dashboard data</p>
              </div>
              <ToggleSwitch
                checked={settings.dashboard.autoRefresh}
                onChange={(value) => handleSettingChange('dashboard', 'autoRefresh', value)}
              />
            </div>
          </div>
        </div>

        {/* System Settings */}
        <div className="card lg:col-span-2">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">⚙️ System Information</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="text-sm font-medium text-gray-900">Theme</label>
              <select
                value={settings.system.theme}
                onChange={(e) => handleSettingChange('system', 'theme', e.target.value)}
                className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              >
                <option value="light">Light</option>
                <option value="dark">Dark (Coming Soon)</option>
              </select>
            </div>

            <div>
              <label className="text-sm font-medium text-gray-900">Application Version</label>
              <div className="mt-1 px-3 py-2 bg-gray-50 border border-gray-300 rounded-lg text-gray-700">
                {settings.system.appVersion}
              </div>
            </div>

            <div>
              <label className="text-sm font-medium text-gray-900">Environment</label>
              <div className="mt-1 px-3 py-2 bg-gray-50 border border-gray-300 rounded-lg text-gray-700">
                {settings.system.environment}
              </div>
            </div>

            <div>
              <label className="text-sm font-medium text-gray-900">API Endpoint</label>
              <div className="mt-1 px-3 py-2 bg-gray-50 border border-gray-300 rounded-lg text-gray-700 truncate text-xs" title={settings.system.apiUrl}>
                {settings.system.apiUrl}
              </div>
            </div>
          </div>
        </div>

        {/* Help & Support */}
        <div className="card lg:col-span-2 bg-blue-50 border-blue-200">
          <div className="flex items-start gap-4">
            <Info className="h-6 w-6 text-blue-600 flex-shrink-0 mt-1" />
            <div>
              <h3 className="font-semibold text-blue-900">Need help?</h3>
              <p className="text-sm text-blue-800 mt-1">
                Check out the documentation or contact our support team at support@constructionai.com
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Settings;
