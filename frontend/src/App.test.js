import { render, screen } from '@testing-library/react';
import App from './App';

test('renders AI Chatbot', () => {
  render(<App />);
  const loadingElement = screen.getByText(/AI Chatbot/i);
  expect(loadingElement).toBeInTheDocument();
});