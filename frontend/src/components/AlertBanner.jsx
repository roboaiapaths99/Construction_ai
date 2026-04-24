import React from 'react';
import { AlertCircle, CheckCircle, XCircle, Info, X } from 'lucide-react';

const AlertBanner = ({ type = 'info', message, onClose, dismissible = false }) => {
  const alertStyles = {
    info: 'bg-blue-50 border-blue-200 text-blue-800',
    success: 'bg-green-50 border-green-200 text-green-800',
    warning: 'bg-yellow-50 border-yellow-200 text-yellow-800',
    error: 'bg-red-50 border-red-200 text-red-800',
  };

  const icons = {
    info: Info,
    success: CheckCircle,
    warning: AlertCircle,
    error: XCircle,
  };

  const Icon = icons[type];

  return (
    <div className={`border rounded-lg p-4 flex items-center justify-between ${alertStyles[type]}`}>
      <div className="flex items-center">
        <Icon className="h-5 w-5 mr-2" />
        <span className="text-sm font-medium">{message}</span>
      </div>
      {dismissible && onClose && (
        <button
          onClick={onClose}
          className="ml-4 text-current hover:opacity-75 transition-opacity"
        >
          <X className="h-4 w-4" />
        </button>
      )}
    </div>
  );
};

export default AlertBanner;
