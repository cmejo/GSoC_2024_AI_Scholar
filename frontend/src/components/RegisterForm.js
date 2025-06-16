import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { authService } from '../services/authService';

function RegisterForm({ onSwitchToLogin, onRegisterSuccess }) {
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    confirmPassword: '',
    firstName: '',
    lastName: '',
    acceptTerms: false
  });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [validationErrors, setValidationErrors] = useState({});
  const [passwordStrength, setPasswordStrength] = useState(0);
  const [isCheckingUsername, setIsCheckingUsername] = useState(false);
  const [isCheckingEmail, setIsCheckingEmail] = useState(false);
  const [usernameAvailable, setUsernameAvailable] = useState(null);
  const [emailAvailable, setEmailAvailable] = useState(null);
  const [currentStep, setCurrentStep] = useState(1); // Multi-step form
  const [isSubmitted, setIsSubmitted] = useState(false);
  
  const { register, isLoading, error, clearError } = useAuth();

  // Password strength calculation
  useEffect(() => {
    if (formData.password) {
      const strength = calculatePasswordStrength(formData.password);
      setPasswordStrength(strength);
    } else {
      setPasswordStrength(0);
    }
  }, [formData.password]);

  // Real-time validation
  useEffect(() => {
    if (isSubmitted) {
      const errors = validateForm();
      setValidationErrors(errors);
    }
  }, [formData, isSubmitted]);

  const calculatePasswordStrength = (password) => {
    let strength = 0;
    if (password.length >= 8) strength += 1;
    if (password.match(/[a-z]/)) strength += 1;
    if (password.match(/[A-Z]/)) strength += 1;
    if (password.match(/[0-9]/)) strength += 1;
    if (password.match(/[^a-zA-Z0-9]/)) strength += 1;
    return strength;
  };

  const getPasswordStrengthText = (strength) => {
    switch (strength) {
      case 0:
      case 1: return { text: 'Very Weak', color: 'text-red-600' };
      case 2: return { text: 'Weak', color: 'text-orange-600' };
      case 3: return { text: 'Fair', color: 'text-yellow-600' };
      case 4: return { text: 'Good', color: 'text-blue-600' };
      case 5: return { text: 'Strong', color: 'text-green-600' };
      default: return { text: '', color: '' };
    }
  };

  const checkUsernameAvailability = async (username) => {
    if (username.length < 3) return;
    
    setIsCheckingUsername(true);
    try {
      const available = await authService.checkUsernameAvailability(username);
      setUsernameAvailable(available);
    } catch (error) {
      console.error('Error checking username:', error);
      setUsernameAvailable(null);
    } finally {
      setIsCheckingUsername(false);
    }
  };

  const checkEmailAvailability = async (email) => {
    if (!/\S+@\S+\.\S+/.test(email)) return;
    
    setIsCheckingEmail(true);
    try {
      const available = await authService.checkEmailAvailability(email);
      setEmailAvailable(available);
    } catch (error) {
      console.error('Error checking email:', error);
      setEmailAvailable(null);
    } finally {
      setIsCheckingEmail(false);
    }
  };

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    const newValue = type === 'checkbox' ? checked : value;
    
    setFormData(prev => ({
      ...prev,
      [name]: newValue
    }));

    // Clear specific validation error
    if (validationErrors[name]) {
      setValidationErrors(prev => ({
        ...prev,
        [name]: ''
      }));
    }

    // Clear global error when user starts typing
    if (error) {
      clearError();
    }

    // Check availability for username and email
    if (name === 'username' && value.length >= 3) {
      const timeoutId = setTimeout(() => checkUsernameAvailability(value), 300);
      return () => clearTimeout(timeoutId);
    }
    
    if (name === 'email' && /\S+@\S+\.\S+/.test(value)) {
      const timeoutId = setTimeout(() => checkEmailAvailability(value), 300);
      return () => clearTimeout(timeoutId);
    }
  };

  const validateForm = () => {
    const errors = {};

    // Step 1 validation (Personal Info)
    if (currentStep >= 1) {
      if (!formData.firstName.trim()) {
        errors.firstName = 'First name is required';
      }
      if (!formData.lastName.trim()) {
        errors.lastName = 'Last name is required';
      }
    }

    // Step 2 validation (Account Info)
    if (currentStep >= 2) {
      if (!formData.username.trim()) {
        errors.username = 'Username is required';
      } else if (formData.username.length < 3) {
        errors.username = 'Username must be at least 3 characters long';
      } else if (!/^[a-zA-Z0-9_]+$/.test(formData.username)) {
        errors.username = 'Username can only contain letters, numbers, and underscores';
      } else if (usernameAvailable === false) {
        errors.username = 'Username is already taken';
      }

      if (!formData.email.trim()) {
        errors.email = 'Email is required';
      } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
        errors.email = 'Please enter a valid email address';
      } else if (emailAvailable === false) {
        errors.email = 'Email is already registered';
      }
    }

    // Step 3 validation (Security)
    if (currentStep >= 3) {
      if (!formData.password) {
        errors.password = 'Password is required';
      } else if (formData.password.length < 8) {
        errors.password = 'Password must be at least 8 characters long';
      } else if (passwordStrength < 3) {
        errors.password = 'Password is too weak. Please use a stronger password.';
      }

      if (!formData.confirmPassword) {
        errors.confirmPassword = 'Please confirm your password';
      } else if (formData.password !== formData.confirmPassword) {
        errors.confirmPassword = 'Passwords do not match';
      }

      if (!formData.acceptTerms) {
        errors.acceptTerms = 'You must accept the terms and conditions';
      }
    }

    return errors;
  };

  const validateCurrentStep = () => {
    const errors = validateForm();
    const stepErrors = {};

    if (currentStep === 1) {
      if (errors.firstName) stepErrors.firstName = errors.firstName;
      if (errors.lastName) stepErrors.lastName = errors.lastName;
    } else if (currentStep === 2) {
      if (errors.username) stepErrors.username = errors.username;
      if (errors.email) stepErrors.email = errors.email;
    } else if (currentStep === 3) {
      if (errors.password) stepErrors.password = errors.password;
      if (errors.confirmPassword) stepErrors.confirmPassword = errors.confirmPassword;
      if (errors.acceptTerms) stepErrors.acceptTerms = errors.acceptTerms;
    }

    return stepErrors;
  };

  const canProceedToNextStep = () => {
    const stepErrors = validateCurrentStep();
    return Object.keys(stepErrors).length === 0;
  };

  const handleNextStep = () => {
    setIsSubmitted(true);
    const stepErrors = validateCurrentStep();
    setValidationErrors(stepErrors);

    if (Object.keys(stepErrors).length === 0) {
      setCurrentStep(prev => Math.min(prev + 1, 3));
      setIsSubmitted(false);
    }
  };

  const handlePrevStep = () => {
    setCurrentStep(prev => Math.max(prev - 1, 1));
    setIsSubmitted(false);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (currentStep < 3) {
      handleNextStep();
      return;
    }

    setIsSubmitted(true);
    const errors = validateForm();
    setValidationErrors(errors);
    
    if (Object.keys(errors).length > 0) {
      return;
    }

    try {
      const additionalData = {
        firstName: formData.firstName,
        lastName: formData.lastName
      };
      await register(formData.username, formData.email, formData.password, additionalData);
      if (onRegisterSuccess) {
        onRegisterSuccess();
      }
    } catch (error) {
      // Error is handled by the auth context
    }
  };

  const getStepTitle = () => {
    switch (currentStep) {
      case 1: return 'Personal Information';
      case 2: return 'Account Details';
      case 3: return 'Security & Terms';
      default: return 'Create Account';
    }
  };

  const getStepDescription = () => {
    switch (currentStep) {
      case 1: return 'Tell us a bit about yourself';
      case 2: return 'Choose your username and email';
      case 3: return 'Set up your password and review terms';
      default: return '';
    }
  };

  return (
    <div className="w-full max-w-lg mx-auto">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl p-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="w-16 h-16 bg-primary-600 rounded-full flex items-center justify-center mx-auto mb-4">
            <i className="fas fa-robot text-white text-2xl"></i>
          </div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {getStepTitle()}
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            {getStepDescription()}
          </p>
        </div>

        {/* Progress Indicator */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-2">
            {[1, 2, 3].map((step) => (
              <div
                key={step}
                className={`flex items-center justify-center w-8 h-8 rounded-full text-sm font-medium ${
                  step <= currentStep
                    ? 'bg-primary-600 text-white'
                    : 'bg-gray-200 dark:bg-gray-700 text-gray-500 dark:text-gray-400'
                }`}
              >
                {step < currentStep ? (
                  <i className="fas fa-check text-xs"></i>
                ) : (
                  step
                )}
              </div>
            ))}
          </div>
          <div className="flex">
            {[1, 2].map((step) => (
              <div
                key={step}
                className={`flex-1 h-1 mx-1 rounded ${
                  step < currentStep
                    ? 'bg-primary-600'
                    : 'bg-gray-200 dark:bg-gray-700'
                }`}
              />
            ))}
          </div>
          <div className="flex justify-between mt-2 text-xs text-gray-500 dark:text-gray-400">
            <span>Personal</span>
            <span>Account</span>
            <span>Security</span>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
            <div className="flex items-center">
              <i className="fas fa-exclamation-circle text-red-600 dark:text-red-400 mr-2"></i>
              <span className="text-red-800 dark:text-red-200 text-sm">{error}</span>
            </div>
          </div>
        )}

        {/* Multi-Step Form */}
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Step 1: Personal Information */}
          {currentStep === 1 && (
            <div className="space-y-6">
              {/* First Name */}
              <div>
                <label htmlFor="firstName" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  First Name
                </label>
                <div className="relative">
                  <input
                    type="text"
                    id="firstName"
                    name="firstName"
                    value={formData.firstName}
                    onChange={handleChange}
                    className={`input-field pl-10 ${validationErrors.firstName ? 'border-red-500 focus:ring-red-500' : ''}`}
                    placeholder="Enter your first name"
                    disabled={isLoading}
                  />
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <i className="fas fa-user text-gray-400"></i>
                  </div>
                </div>
                {validationErrors.firstName && (
                  <p className="mt-1 text-xs text-red-600 dark:text-red-400">
                    {validationErrors.firstName}
                  </p>
                )}
              </div>

              {/* Last Name */}
              <div>
                <label htmlFor="lastName" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Last Name
                </label>
                <div className="relative">
                  <input
                    type="text"
                    id="lastName"
                    name="lastName"
                    value={formData.lastName}
                    onChange={handleChange}
                    className={`input-field pl-10 ${validationErrors.lastName ? 'border-red-500 focus:ring-red-500' : ''}`}
                    placeholder="Enter your last name"
                    disabled={isLoading}
                  />
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <i className="fas fa-user text-gray-400"></i>
                  </div>
                </div>
                {validationErrors.lastName && (
                  <p className="mt-1 text-xs text-red-600 dark:text-red-400">
                    {validationErrors.lastName}
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Step 2: Account Details */}
          {currentStep === 2 && (
            <div className="space-y-6">
              {/* Username */}
              <div>
                <label htmlFor="username" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Username
                </label>
                <div className="relative">
                  <input
                    type="text"
                    id="username"
                    name="username"
                    value={formData.username}
                    onChange={handleChange}
                    className={`input-field pl-10 pr-10 ${validationErrors.username ? 'border-red-500 focus:ring-red-500' : ''}`}
                    placeholder="Choose a username"
                    disabled={isLoading}
                  />
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <i className="fas fa-at text-gray-400"></i>
                  </div>
                  <div className="absolute inset-y-0 right-0 pr-3 flex items-center">
                    {isCheckingUsername ? (
                      <div className="w-4 h-4 border-2 border-gray-400 border-t-transparent rounded-full animate-spin"></div>
                    ) : usernameAvailable === true ? (
                      <i className="fas fa-check text-green-500"></i>
                    ) : usernameAvailable === false ? (
                      <i className="fas fa-times text-red-500"></i>
                    ) : null}
                  </div>
                </div>
                {validationErrors.username && (
                  <p className="mt-1 text-xs text-red-600 dark:text-red-400">
                    {validationErrors.username}
                  </p>
                )}
                {!validationErrors.username && usernameAvailable === true && (
                  <p className="mt-1 text-xs text-green-600 dark:text-green-400">
                    Username is available!
                  </p>
                )}
              </div>

              {/* Email */}
              <div>
                <label htmlFor="email" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Email Address
                </label>
                <div className="relative">
                  <input
                    type="email"
                    id="email"
                    name="email"
                    value={formData.email}
                    onChange={handleChange}
                    className={`input-field pl-10 pr-10 ${validationErrors.email ? 'border-red-500 focus:ring-red-500' : ''}`}
                    placeholder="Enter your email address"
                    disabled={isLoading}
                  />
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <i className="fas fa-envelope text-gray-400"></i>
                  </div>
                  <div className="absolute inset-y-0 right-0 pr-3 flex items-center">
                    {isCheckingEmail ? (
                      <div className="w-4 h-4 border-2 border-gray-400 border-t-transparent rounded-full animate-spin"></div>
                    ) : emailAvailable === true ? (
                      <i className="fas fa-check text-green-500"></i>
                    ) : emailAvailable === false ? (
                      <i className="fas fa-times text-red-500"></i>
                    ) : null}
                  </div>
                </div>
                {validationErrors.email && (
                  <p className="mt-1 text-xs text-red-600 dark:text-red-400">
                    {validationErrors.email}
                  </p>
                )}
                {!validationErrors.email && emailAvailable === true && (
                  <p className="mt-1 text-xs text-green-600 dark:text-green-400">
                    Email is available!
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Step 3: Security & Terms */}
          {currentStep === 3 && (
            <div className="space-y-6">
              {/* Password */}
              <div>
                <label htmlFor="password" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Password
                </label>
                <div className="relative">
                  <input
                    type={showPassword ? 'text' : 'password'}
                    id="password"
                    name="password"
                    value={formData.password}
                    onChange={handleChange}
                    className={`input-field pl-10 pr-10 ${validationErrors.password ? 'border-red-500 focus:ring-red-500' : ''}`}
                    placeholder="Create a strong password"
                    disabled={isLoading}
                  />
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <i className="fas fa-lock text-gray-400"></i>
                  </div>
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                    disabled={isLoading}
                  >
                    <i className={`fas ${showPassword ? 'fa-eye-slash' : 'fa-eye'}`}></i>
                  </button>
                </div>
                
                {/* Password Strength Indicator */}
                {formData.password && (
                  <div className="mt-2">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs text-gray-600 dark:text-gray-400">Password Strength</span>
                      <span className={`text-xs font-medium ${getPasswordStrengthText(passwordStrength).color}`}>
                        {getPasswordStrengthText(passwordStrength).text}
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1.5">
                      <div
                        className={`h-1.5 rounded-full transition-all duration-300 ${
                          passwordStrength <= 1 ? 'bg-red-500' :
                          passwordStrength === 2 ? 'bg-orange-500' :
                          passwordStrength === 3 ? 'bg-yellow-500' :
                          passwordStrength === 4 ? 'bg-blue-500' :
                          'bg-green-500'
                        }`}
                        style={{ width: `${(passwordStrength / 5) * 100}%` }}
                      ></div>
                    </div>
                  </div>
                )}
                
                {validationErrors.password && (
                  <p className="mt-1 text-xs text-red-600 dark:text-red-400">
                    {validationErrors.password}
                  </p>
                )}
              </div>

              {/* Confirm Password */}
              <div>
                <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Confirm Password
                </label>
                <div className="relative">
                  <input
                    type={showConfirmPassword ? 'text' : 'password'}
                    id="confirmPassword"
                    name="confirmPassword"
                    value={formData.confirmPassword}
                    onChange={handleChange}
                    className={`input-field pl-10 pr-10 ${validationErrors.confirmPassword ? 'border-red-500 focus:ring-red-500' : ''}`}
                    placeholder="Confirm your password"
                    disabled={isLoading}
                  />
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <i className="fas fa-lock text-gray-400"></i>
                  </div>
                  <button
                    type="button"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                    disabled={isLoading}
                  >
                    <i className={`fas ${showConfirmPassword ? 'fa-eye-slash' : 'fa-eye'}`}></i>
                  </button>
                </div>
                {validationErrors.confirmPassword && (
                  <p className="mt-1 text-xs text-red-600 dark:text-red-400">
                    {validationErrors.confirmPassword}
                  </p>
                )}
                {!validationErrors.confirmPassword && formData.confirmPassword && formData.password === formData.confirmPassword && (
                  <p className="mt-1 text-xs text-green-600 dark:text-green-400">
                    Passwords match!
                  </p>
                )}
              </div>

              {/* Terms and Conditions */}
              <div>
                <label className="flex items-start">
                  <input
                    type="checkbox"
                    name="acceptTerms"
                    checked={formData.acceptTerms}
                    onChange={handleChange}
                    className={`mt-1 w-4 h-4 text-primary-600 bg-gray-100 border-gray-300 rounded focus:ring-primary-500 dark:focus:ring-primary-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600 ${
                      validationErrors.acceptTerms ? 'border-red-500' : ''
                    }`}
                    disabled={isLoading}
                  />
                  <span className="ml-3 text-sm text-gray-600 dark:text-gray-400">
                    I agree to the{' '}
                    <a href="#" className="text-primary-600 hover:text-primary-700 dark:text-primary-400 dark:hover:text-primary-300 underline">
                      Terms of Service
                    </a>{' '}
                    and{' '}
                    <a href="#" className="text-primary-600 hover:text-primary-700 dark:text-primary-400 dark:hover:text-primary-300 underline">
                      Privacy Policy
                    </a>
                  </span>
                </label>
                {validationErrors.acceptTerms && (
                  <p className="mt-1 text-xs text-red-600 dark:text-red-400">
                    {validationErrors.acceptTerms}
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Navigation Buttons */}
          <div className="flex justify-between pt-6">
            {currentStep > 1 && (
              <button
                type="button"
                onClick={handlePrevStep}
                disabled={isLoading}
                className="px-6 py-2 text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 transition-colors"
              >
                <i className="fas fa-arrow-left mr-2"></i>
                Back
              </button>
            )}
            
            <div className="flex-1"></div>
            
            <button
              type="submit"
              disabled={isLoading || (currentStep < 3 && !canProceedToNextStep())}
              className="px-8 py-3 bg-primary-600 hover:bg-primary-700 text-white font-semibold rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <div className="flex items-center">
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                  Creating Account...
                </div>
              ) : currentStep < 3 ? (
                <>
                  Next
                  <i className="fas fa-arrow-right ml-2"></i>
                </>
              ) : (
                'Create Account'
              )}
            </button>
          </div>
        </form>

        {/* Switch to Login */}
        <div className="mt-8 text-center">
          <p className="text-gray-600 dark:text-gray-400">
            Already have an account?{' '}
            <button
              onClick={onSwitchToLogin}
              className="text-primary-600 hover:text-primary-700 dark:text-primary-400 dark:hover:text-primary-300 font-medium"
              disabled={isLoading}
            >
              Sign in
            </button>
          </p>
        </div>
      </div>
    </div>
  );
}

export default RegisterForm;