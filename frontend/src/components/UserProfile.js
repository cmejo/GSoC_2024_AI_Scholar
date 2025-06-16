import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { authService } from '../services/authService';

function UserProfile({ onClose, onShowToast }) {
  const { user, token, logout } = useAuth();
  const [activeTab, setActiveTab] = useState('profile');
  const [isEditing, setIsEditing] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isChangingPassword, setIsChangingPassword] = useState(false);
  const [profileData, setProfileData] = useState({
    firstName: '',
    lastName: '',
    email: '',
    username: '',
    bio: '',
    avatar: null
  });
  const [passwordData, setPasswordData] = useState({
    currentPassword: '',
    newPassword: '',
    confirmPassword: ''
  });
  const [validationErrors, setValidationErrors] = useState({});
  const [userStats, setUserStats] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  useEffect(() => {
    if (user) {
      setProfileData({
        firstName: user.firstName || '',
        lastName: user.lastName || '',
        email: user.email || '',
        username: user.username || '',
        bio: user.bio || '',
        avatar: user.avatar || null
      });
      loadUserStats();
      loadUserSessions();
    }
  }, [user]);

  useEffect(() => {
    // Close modal on Escape key
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  const loadUserStats = async () => {
    try {
      const stats = await authService.getUserStats(token);
      setUserStats(stats);
    } catch (error) {
      console.error('Failed to load user stats:', error);
    }
  };

  const loadUserSessions = async () => {
    try {
      const userSessions = await authService.getUserSessions(token);
      setSessions(userSessions);
    } catch (error) {
      console.error('Failed to load user sessions:', error);
    }
  };

  const handleProfileChange = (e) => {
    const { name, value } = e.target;
    setProfileData(prev => ({
      ...prev,
      [name]: value
    }));

    // Clear validation error for this field
    if (validationErrors[name]) {
      setValidationErrors(prev => ({
        ...prev,
        [name]: ''
      }));
    }
  };

  const handlePasswordChange = (e) => {
    const { name, value } = e.target;
    setPasswordData(prev => ({
      ...prev,
      [name]: value
    }));

    // Clear validation error for this field
    if (validationErrors[name]) {
      setValidationErrors(prev => ({
        ...prev,
        [name]: ''
      }));
    }
  };

  const validateProfile = () => {
    const errors = {};

    if (!profileData.firstName.trim()) {
      errors.firstName = 'First name is required';
    }

    if (!profileData.lastName.trim()) {
      errors.lastName = 'Last name is required';
    }

    if (!profileData.email.trim()) {
      errors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(profileData.email)) {
      errors.email = 'Please enter a valid email address';
    }

    if (!profileData.username.trim()) {
      errors.username = 'Username is required';
    } else if (profileData.username.length < 3) {
      errors.username = 'Username must be at least 3 characters long';
    }

    return errors;
  };

  const validatePassword = () => {
    const errors = {};

    if (!passwordData.currentPassword) {
      errors.currentPassword = 'Current password is required';
    }

    if (!passwordData.newPassword) {
      errors.newPassword = 'New password is required';
    } else if (passwordData.newPassword.length < 8) {
      errors.newPassword = 'Password must be at least 8 characters long';
    }

    if (!passwordData.confirmPassword) {
      errors.confirmPassword = 'Please confirm your new password';
    } else if (passwordData.newPassword !== passwordData.confirmPassword) {
      errors.confirmPassword = 'Passwords do not match';
    }

    return errors;
  };

  const handleSaveProfile = async () => {
    const errors = validateProfile();
    setValidationErrors(errors);

    if (Object.keys(errors).length > 0) {
      return;
    }

    setIsSaving(true);
    try {
      await authService.updateProfile(profileData, token);
      setIsEditing(false);
      onShowToast('Profile updated successfully', 'success');
    } catch (error) {
      onShowToast(error.message || 'Failed to update profile', 'error');
    } finally {
      setIsSaving(false);
    }
  };

  const handleChangePassword = async () => {
    const errors = validatePassword();
    setValidationErrors(errors);

    if (Object.keys(errors).length > 0) {
      return;
    }

    setIsChangingPassword(true);
    try {
      await authService.changePassword(passwordData, token);
      setPasswordData({
        currentPassword: '',
        newPassword: '',
        confirmPassword: ''
      });
      onShowToast('Password changed successfully', 'success');
    } catch (error) {
      onShowToast(error.message || 'Failed to change password', 'error');
    } finally {
      setIsChangingPassword(false);
    }
  };

  const handleDeleteAccount = async () => {
    try {
      await authService.deleteAccount(token);
      onShowToast('Account deleted successfully', 'success');
      logout();
    } catch (error) {
      onShowToast(error.message || 'Failed to delete account', 'error');
    }
  };

  const handleLogoutAllSessions = async () => {
    try {
      await authService.logoutAllSessions(token);
      onShowToast('Logged out from all sessions', 'success');
      logout();
    } catch (error) {
      onShowToast(error.message || 'Failed to logout from all sessions', 'error');
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getInitials = (firstName, lastName) => {
    return `${firstName?.charAt(0) || ''}${lastName?.charAt(0) || ''}`.toUpperCase();
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
            User Profile
          </h2>
          <button
            onClick={onClose}
            className="p-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
          >
            <i className="fas fa-times text-lg"></i>
          </button>
        </div>

        {/* Tabs */}
        <div className="border-b border-gray-200 dark:border-gray-700">
          <nav className="flex space-x-8 px-6">
            {[
              { id: 'profile', label: 'Profile', icon: 'fa-user' },
              { id: 'security', label: 'Security', icon: 'fa-shield-alt' },
              { id: 'stats', label: 'Statistics', icon: 'fa-chart-bar' },
              { id: 'sessions', label: 'Sessions', icon: 'fa-desktop' }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab.id
                    ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                    : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:border-gray-300'
                }`}
              >
                <i className={`fas ${tab.icon} mr-2`}></i>
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[calc(90vh-200px)]">
          {/* Profile Tab */}
          {activeTab === 'profile' && (
            <div className="space-y-6">
              {/* Avatar Section */}
              <div className="flex items-center space-x-6">
                <div className="relative">
                  <div className="w-24 h-24 bg-primary-600 rounded-full flex items-center justify-center text-white text-2xl font-bold">
                    {profileData.avatar ? (
                      <img
                        src={profileData.avatar}
                        alt="Profile"
                        className="w-24 h-24 rounded-full object-cover"
                      />
                    ) : (
                      getInitials(profileData.firstName, profileData.lastName)
                    )}
                  </div>
                  {isEditing && (
                    <button className="absolute bottom-0 right-0 w-8 h-8 bg-primary-600 text-white rounded-full flex items-center justify-center hover:bg-primary-700 transition-colors">
                      <i className="fas fa-camera text-sm"></i>
                    </button>
                  )}
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                    {profileData.firstName} {profileData.lastName}
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400">@{profileData.username}</p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Member since {formatDate(user?.created_at)}
                  </p>
                </div>
              </div>

              {/* Profile Form */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    First Name
                  </label>
                  <input
                    type="text"
                    name="firstName"
                    value={profileData.firstName}
                    onChange={handleProfileChange}
                    disabled={!isEditing}
                    className={`input-field ${validationErrors.firstName ? 'border-red-500' : ''} ${
                      !isEditing ? 'bg-gray-50 dark:bg-gray-700' : ''
                    }`}
                  />
                  {validationErrors.firstName && (
                    <p className="mt-1 text-xs text-red-600 dark:text-red-400">
                      {validationErrors.firstName}
                    </p>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Last Name
                  </label>
                  <input
                    type="text"
                    name="lastName"
                    value={profileData.lastName}
                    onChange={handleProfileChange}
                    disabled={!isEditing}
                    className={`input-field ${validationErrors.lastName ? 'border-red-500' : ''} ${
                      !isEditing ? 'bg-gray-50 dark:bg-gray-700' : ''
                    }`}
                  />
                  {validationErrors.lastName && (
                    <p className="mt-1 text-xs text-red-600 dark:text-red-400">
                      {validationErrors.lastName}
                    </p>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Username
                  </label>
                  <input
                    type="text"
                    name="username"
                    value={profileData.username}
                    onChange={handleProfileChange}
                    disabled={!isEditing}
                    className={`input-field ${validationErrors.username ? 'border-red-500' : ''} ${
                      !isEditing ? 'bg-gray-50 dark:bg-gray-700' : ''
                    }`}
                  />
                  {validationErrors.username && (
                    <p className="mt-1 text-xs text-red-600 dark:text-red-400">
                      {validationErrors.username}
                    </p>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Email
                  </label>
                  <input
                    type="email"
                    name="email"
                    value={profileData.email}
                    onChange={handleProfileChange}
                    disabled={!isEditing}
                    className={`input-field ${validationErrors.email ? 'border-red-500' : ''} ${
                      !isEditing ? 'bg-gray-50 dark:bg-gray-700' : ''
                    }`}
                  />
                  {validationErrors.email && (
                    <p className="mt-1 text-xs text-red-600 dark:text-red-400">
                      {validationErrors.email}
                    </p>
                  )}
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Bio
                </label>
                <textarea
                  name="bio"
                  value={profileData.bio}
                  onChange={handleProfileChange}
                  disabled={!isEditing}
                  rows={3}
                  className={`input-field ${!isEditing ? 'bg-gray-50 dark:bg-gray-700' : ''}`}
                  placeholder="Tell us about yourself..."
                />
              </div>

              {/* Action Buttons */}
              <div className="flex justify-end space-x-3">
                {isEditing ? (
                  <>
                    <button
                      onClick={() => {
                        setIsEditing(false);
                        setValidationErrors({});
                      }}
                      disabled={isSaving}
                      className="px-4 py-2 text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 transition-colors"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleSaveProfile}
                      disabled={isSaving}
                      className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors disabled:opacity-50"
                    >
                      {isSaving ? (
                        <div className="flex items-center">
                          <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                          Saving...
                        </div>
                      ) : (
                        'Save Changes'
                      )}
                    </button>
                  </>
                ) : (
                  <button
                    onClick={() => setIsEditing(true)}
                    className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors"
                  >
                    <i className="fas fa-edit mr-2"></i>
                    Edit Profile
                  </button>
                )}
              </div>
            </div>
          )}

          {/* Security Tab */}
          {activeTab === 'security' && (
            <div className="space-y-8">
              {/* Change Password */}
              <div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
                  Change Password
                </h3>
                <div className="space-y-4 max-w-md">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Current Password
                    </label>
                    <input
                      type="password"
                      name="currentPassword"
                      value={passwordData.currentPassword}
                      onChange={handlePasswordChange}
                      className={`input-field ${validationErrors.currentPassword ? 'border-red-500' : ''}`}
                    />
                    {validationErrors.currentPassword && (
                      <p className="mt-1 text-xs text-red-600 dark:text-red-400">
                        {validationErrors.currentPassword}
                      </p>
                    )}
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      New Password
                    </label>
                    <input
                      type="password"
                      name="newPassword"
                      value={passwordData.newPassword}
                      onChange={handlePasswordChange}
                      className={`input-field ${validationErrors.newPassword ? 'border-red-500' : ''}`}
                    />
                    {validationErrors.newPassword && (
                      <p className="mt-1 text-xs text-red-600 dark:text-red-400">
                        {validationErrors.newPassword}
                      </p>
                    )}
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Confirm New Password
                    </label>
                    <input
                      type="password"
                      name="confirmPassword"
                      value={passwordData.confirmPassword}
                      onChange={handlePasswordChange}
                      className={`input-field ${validationErrors.confirmPassword ? 'border-red-500' : ''}`}
                    />
                    {validationErrors.confirmPassword && (
                      <p className="mt-1 text-xs text-red-600 dark:text-red-400">
                        {validationErrors.confirmPassword}
                      </p>
                    )}
                  </div>

                  <button
                    onClick={handleChangePassword}
                    disabled={isChangingPassword}
                    className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors disabled:opacity-50"
                  >
                    {isChangingPassword ? (
                      <div className="flex items-center">
                        <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                        Changing...
                      </div>
                    ) : (
                      'Change Password'
                    )}
                  </button>
                </div>
              </div>

              {/* Security Actions */}
              <div className="border-t border-gray-200 dark:border-gray-700 pt-8">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
                  Security Actions
                </h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-4 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg">
                    <div>
                      <h4 className="font-medium text-yellow-800 dark:text-yellow-200">
                        Logout from all sessions
                      </h4>
                      <p className="text-sm text-yellow-600 dark:text-yellow-300">
                        This will log you out from all devices and browsers
                      </p>
                    </div>
                    <button
                      onClick={handleLogoutAllSessions}
                      className="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 text-white rounded-lg transition-colors"
                    >
                      Logout All
                    </button>
                  </div>

                  <div className="flex items-center justify-between p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                    <div>
                      <h4 className="font-medium text-red-800 dark:text-red-200">
                        Delete Account
                      </h4>
                      <p className="text-sm text-red-600 dark:text-red-300">
                        Permanently delete your account and all associated data
                      </p>
                    </div>
                    <button
                      onClick={() => setShowDeleteConfirm(true)}
                      className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
                    >
                      Delete Account
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Statistics Tab */}
          {activeTab === 'stats' && (
            <div className="space-y-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                Your Statistics
              </h3>
              
              {userStats ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  <div className="bg-primary-50 dark:bg-primary-900/20 p-6 rounded-lg">
                    <div className="flex items-center">
                      <div className="p-3 bg-primary-600 rounded-lg">
                        <i className="fas fa-comments text-white text-xl"></i>
                      </div>
                      <div className="ml-4">
                        <p className="text-sm text-gray-600 dark:text-gray-400">Total Sessions</p>
                        <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                          {userStats.totalSessions}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-lg">
                    <div className="flex items-center">
                      <div className="p-3 bg-green-600 rounded-lg">
                        <i className="fas fa-comment-dots text-white text-xl"></i>
                      </div>
                      <div className="ml-4">
                        <p className="text-sm text-gray-600 dark:text-gray-400">Total Messages</p>
                        <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                          {userStats.totalMessages}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg">
                    <div className="flex items-center">
                      <div className="p-3 bg-blue-600 rounded-lg">
                        <i className="fas fa-user text-white text-xl"></i>
                      </div>
                      <div className="ml-4">
                        <p className="text-sm text-gray-600 dark:text-gray-400">Your Messages</p>
                        <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                          {userStats.userMessages}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-lg">
                    <div className="flex items-center">
                      <div className="p-3 bg-purple-600 rounded-lg">
                        <i className="fas fa-robot text-white text-xl"></i>
                      </div>
                      <div className="ml-4">
                        <p className="text-sm text-gray-600 dark:text-gray-400">AI Responses</p>
                        <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                          {userStats.botMessages}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex items-center justify-center py-12">
                  <div className="w-8 h-8 border-2 border-primary-600 border-t-transparent rounded-full animate-spin"></div>
                  <span className="ml-3 text-gray-600 dark:text-gray-400">Loading statistics...</span>
                </div>
              )}
            </div>
          )}

          {/* Sessions Tab */}
          {activeTab === 'sessions' && (
            <div className="space-y-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                Active Sessions
              </h3>
              
              {sessions.length > 0 ? (
                <div className="space-y-4">
                  {sessions.map((session) => (
                    <div key={session.id} className="flex items-center justify-between p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
                      <div className="flex items-center space-x-4">
                        <div className="p-2 bg-gray-100 dark:bg-gray-700 rounded-lg">
                          <i className="fas fa-desktop text-gray-600 dark:text-gray-400"></i>
                        </div>
                        <div>
                          <p className="font-medium text-gray-900 dark:text-gray-100">
                            {session.userAgent || 'Unknown Device'}
                          </p>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            {session.ipAddress} • Last active: {formatDate(session.lastActivity)}
                          </p>
                          {session.isCurrent && (
                            <span className="inline-block px-2 py-1 bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-200 text-xs rounded-full">
                              Current Session
                            </span>
                          )}
                        </div>
                      </div>
                      {!session.isCurrent && (
                        <button className="px-3 py-1 text-red-600 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300 text-sm">
                          Revoke
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <i className="fas fa-desktop text-4xl text-gray-400 dark:text-gray-500 mb-4"></i>
                  <p className="text-gray-600 dark:text-gray-400">No active sessions found</p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Delete Account Confirmation Modal */}
        {showDeleteConfirm && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-md w-full p-6">
              <div className="flex items-center mb-4">
                <div className="w-12 h-12 bg-red-100 dark:bg-red-900/20 rounded-full flex items-center justify-center">
                  <i className="fas fa-exclamation-triangle text-red-600 dark:text-red-400 text-xl"></i>
                </div>
                <div className="ml-4">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                    Delete Account
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    This action cannot be undone
                  </p>
                </div>
              </div>
              
              <p className="text-gray-700 dark:text-gray-300 mb-6">
                Are you sure you want to delete your account? All your data, including chat history and settings, will be permanently removed.
              </p>
              
              <div className="flex justify-end space-x-3">
                <button
                  onClick={() => setShowDeleteConfirm(false)}
                  className="px-4 py-2 text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleDeleteAccount}
                  className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
                >
                  Delete Account
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default UserProfile;