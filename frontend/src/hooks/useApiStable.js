import { useState, useEffect, useCallback, useRef } from 'react';
import { apiEndpoints } from '../api';

export const useApiStable = (apiFunction, dependencies = [], preventRerender = false) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const lastUpdateRef = useRef(Date.now());
  const dataRef = useRef(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiFunction();
      const newData = response.data;
      
      // Only update state if data actually changed or rerender is allowed
      if (!preventRerender || JSON.stringify(newData) !== JSON.stringify(dataRef.current)) {
        setData(newData);
        dataRef.current = newData;
        lastUpdateRef.current = Date.now();
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  }, dependencies);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const refetch = useCallback(() => {
    fetchData();
  }, [fetchData]);

  return { data, loading, error, refetch };
};

export const usePollingStable = (apiFunction, interval = 30000, dependencies = [], preventRerender = false) => {
  const { data, loading, error, refetch } = useApiStable(apiFunction, dependencies, preventRerender);
  const intervalRef = useRef(null);
  const isWebcamActiveRef = useRef(false);

  // Function to check if webcam is active (you can pass this as a prop)
  const setWebcamActive = useCallback((active) => {
    isWebcamActiveRef.current = active;
  }, []);

  useEffect(() => {
    const startPolling = () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      
      intervalRef.current = setInterval(() => {
        // Don't poll if webcam is active and preventRerender is true
        if (!preventRerender || !isWebcamActiveRef.current) {
          refetch();
        }
      }, interval);
    };

    startPolling();

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [refetch, interval, preventRerender]);

  return { data, loading, error, refetch, setWebcamActive };
};

export const useConditionalPolling = (apiFunction, interval = 30000, dependencies = [], condition = true) => {
  const { data, loading, error, refetch } = useApiStable(apiFunction, dependencies);
  const intervalRef = useRef(null);

  useEffect(() => {
    if (condition) {
      intervalRef.current = setInterval(() => {
        refetch();
      }, interval);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [refetch, interval, condition]);

  return { data, loading, error, refetch };
};
