import React from 'react';
import './Loading.css';

interface LoadingProps {
  show: boolean;
}

const Loading: React.FC<LoadingProps> = ({ show }) => {
  if (!show) return null;

  return (
    <div className="loading-overlay">
      <div className="loading-spinner">⏳</div>
    </div>
  );
};

export default Loading;
