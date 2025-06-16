# GitHub Actions Workflows

This document describes the automated workflows set up for the AI Chatbot Web GUI project.

## 🔄 Continuous Integration (CI)

**File:** `.github/workflows/ci.yml`

### Triggers
- Push to `main` and `develop` branches
- Pull requests to `main` and `develop` branches

### Jobs

#### 1. **Test Suite** (`test`)
- **Matrix Testing**: Python 3.8-3.12 × Node.js 18,20
- **Code Quality Checks**:
  - Black formatting
  - isort import sorting
  - flake8 linting
  - bandit security scanning
  - safety dependency checking
- **Testing**:
  - Python unit tests with pytest
  - Coverage reporting
  - Application startup test
- **Frontend Testing**:
  - ESLint for JavaScript
  - Lighthouse CI for performance

#### 2. **Security Scan** (`security-scan`)
- Trivy vulnerability scanner
- SARIF upload to GitHub Security tab
- Dependency review for PRs

#### 3. **Docker Test** (`docker-test`)
- Docker image build test
- Container functionality test
- Multi-stage build validation

#### 4. **Performance Test** (`performance-test`)
- Locust load testing
- Performance report generation
- Memory usage analysis

#### 5. **Accessibility Test** (`accessibility-test`)
- Lighthouse accessibility audit
- WCAG compliance checking
- Mobile accessibility testing

## 🚀 Deployment

**File:** `.github/workflows/deploy.yml`

### Triggers
- Git tags matching `v*` pattern
- GitHub releases

### Jobs

#### 1. **Build and Test** (`build-and-test`)
- Final validation before deployment
- Comprehensive test suite execution

#### 2. **Docker Build and Push** (`docker-build-and-push`)
- Multi-platform Docker image build
- Push to Docker Hub with semantic versioning
- Image vulnerability scanning

#### 3. **Deploy to Heroku** (`deploy-to-heroku`)
- Automated Heroku deployment
- Environment variable configuration
- Health check validation

#### 4. **Deploy to Railway** (`deploy-to-railway`)
- Railway platform deployment
- Automatic scaling configuration

#### 5. **Create GitHub Release** (`create-github-release`)
- Automated changelog generation
- Release notes with installation instructions
- Asset attachment

#### 6. **Notify Deployment** (`notify-deployment`)
- Slack/Discord notifications
- Deployment status reporting

## 🔍 Code Quality

**File:** `.github/workflows/code-quality.yml`

### Triggers
- Push to `main` and `develop` branches
- Pull requests
- Weekly scheduled runs (Sundays 2 AM UTC)

### Jobs

#### 1. **Code Quality Analysis** (`code-quality`)
- **Formatting**: Black, isort
- **Linting**: flake8, pylint
- **Type Checking**: mypy
- **Security**: bandit, safety
- **Complexity**: radon analysis
- **Maintainability**: radon MI

#### 2. **SonarCloud Analysis** (`sonarcloud`)
- Code quality metrics
- Technical debt analysis
- Security hotspots
- Coverage analysis

#### 3. **CodeQL Analysis** (`codeql`)
- GitHub's semantic code analysis
- Security vulnerability detection
- Python and JavaScript analysis

#### 4. **Dependency Review** (`dependency-review`)
- PR dependency impact analysis
- License compatibility checking
- Security vulnerability detection

#### 5. **License Check** (`license-check`)
- Dependency license validation
- GPL license detection
- License report generation

#### 6. **Documentation Check** (`documentation-check`)
- Required file validation
- Markdown linting
- Broken link detection

#### 7. **Performance Analysis** (`performance-analysis`)
- Memory profiling
- Performance regression detection

## 🔄 Auto Update

**File:** `.github/workflows/auto-update.yml`

### Triggers
- Weekly scheduled runs (Mondays 3 AM UTC)
- Manual workflow dispatch

### Jobs

#### 1. **Update Dependencies** (`update-dependencies`)
- Python package updates
- Automated testing with new versions
- Pull request creation

#### 2. **Update GitHub Actions** (`update-github-actions`)
- Action version updates
- Renovate bot integration

#### 3. **Security Updates** (`security-updates`)
- Vulnerability scanning
- Security issue creation
- Priority labeling

#### 4. **Update Documentation** (`update-documentation`)
- API documentation generation
- Badge updates
- Version synchronization

#### 5. **Cleanup Old Artifacts** (`cleanup-old-artifacts`)
- 30-day artifact retention
- Storage optimization

#### 6. **Repository Health Check** (`health-check`)
- Large file detection
- TODO/FIXME tracking
- Code statistics

## 🔧 Configuration Files

### Workflow Configuration
- **Renovate**: `.github/renovate.json` - Dependency updates
- **Markdown Link Check**: `.github/mlc_config.json` - Link validation
- **SonarCloud**: `sonar-project.properties` - Code analysis

### Code Quality
- **Python Project**: `pyproject.toml` - Project metadata and tool config
- **Pytest**: `pytest.ini` - Test configuration
- **Pre-commit**: `.pre-commit-config.yaml` - Git hooks

### Development
- **Requirements**: `requirements-dev.txt` - Development dependencies
- **Docker**: `Dockerfile` - Multi-stage container build

## 📊 Workflow Status Badges

Add these badges to your README.md:

```markdown
[![CI](https://github.com/YOUR_USERNAME/ai-chatbot-web-gui/workflows/Continuous%20Integration/badge.svg)](https://github.com/YOUR_USERNAME/ai-chatbot-web-gui/actions/workflows/ci.yml)
[![Deploy](https://github.com/YOUR_USERNAME/ai-chatbot-web-gui/workflows/Deploy/badge.svg)](https://github.com/YOUR_USERNAME/ai-chatbot-web-gui/actions/workflows/deploy.yml)
[![Code Quality](https://github.com/YOUR_USERNAME/ai-chatbot-web-gui/workflows/Code%20Quality/badge.svg)](https://github.com/YOUR_USERNAME/ai-chatbot-web-gui/actions/workflows/code-quality.yml)
[![Security](https://github.com/YOUR_USERNAME/ai-chatbot-web-gui/workflows/Security/badge.svg)](https://github.com/YOUR_USERNAME/ai-chatbot-web-gui/actions/workflows/security.yml)
```

## 🔐 Required Secrets

Configure these secrets in your GitHub repository settings:

### Docker Hub
- `DOCKER_USERNAME` - Docker Hub username
- `DOCKER_PASSWORD` - Docker Hub password/token

### Heroku Deployment
- `HEROKU_API_KEY` - Heroku API key
- `HEROKU_APP_NAME` - Heroku application name
- `HEROKU_EMAIL` - Heroku account email

### Railway Deployment
- `RAILWAY_TOKEN` - Railway API token
- `RAILWAY_SERVICE_ID` - Railway service ID

### Code Analysis
- `SONAR_TOKEN` - SonarCloud token
- `CODECOV_TOKEN` - Codecov token (optional)

### Notifications
- `SLACK_WEBHOOK_URL` - Slack webhook for notifications
- `DISCORD_WEBHOOK` - Discord webhook for notifications

### Application
- `SECRET_KEY` - Flask secret key for production
- `OLLAMA_BASE_URL` - Ollama server URL for deployment

## 🎯 Workflow Features

### Automated Testing
- **Multi-version testing** across Python 3.8-3.12
- **Cross-platform testing** on Ubuntu, Windows, macOS
- **Browser compatibility** testing
- **Mobile responsiveness** validation

### Security
- **Dependency vulnerability** scanning
- **Code security** analysis with bandit
- **Container security** with Trivy
- **Secret detection** with detect-secrets

### Performance
- **Load testing** with Locust
- **Memory profiling** analysis
- **Lighthouse performance** audits
- **Bundle size** monitoring

### Quality Assurance
- **Code coverage** tracking (80% minimum)
- **Code complexity** analysis
- **Maintainability** scoring
- **Documentation** completeness

### Automation
- **Dependency updates** with Renovate
- **Security patches** auto-detection
- **Release automation** with semantic versioning
- **Deployment** to multiple platforms

## 🚀 Getting Started

1. **Fork the repository**
2. **Configure secrets** in your repository settings
3. **Customize workflows** for your needs:
   - Update repository URLs
   - Modify deployment targets
   - Adjust notification settings
4. **Push changes** to trigger workflows
5. **Monitor results** in the Actions tab

## 🔧 Customization

### Adding New Workflows
1. Create `.github/workflows/your-workflow.yml`
2. Define triggers and jobs
3. Add required secrets
4. Test with a pull request

### Modifying Existing Workflows
1. Edit workflow files in `.github/workflows/`
2. Update configuration files as needed
3. Test changes in a feature branch
4. Monitor workflow runs

### Environment-Specific Configuration
- **Development**: Use `develop` branch triggers
- **Staging**: Add staging deployment jobs
- **Production**: Use tag-based triggers

## 📈 Monitoring and Maintenance

### Regular Tasks
- **Review workflow runs** weekly
- **Update dependencies** monthly
- **Check security alerts** immediately
- **Monitor performance** trends

### Troubleshooting
- **Check workflow logs** for failures
- **Verify secrets** are correctly set
- **Test locally** before pushing
- **Review PR checks** before merging

### Optimization
- **Cache dependencies** for faster builds
- **Parallelize jobs** where possible
- **Use matrix strategies** for multi-version testing
- **Optimize Docker** builds with multi-stage

This comprehensive GitHub Actions setup ensures high code quality, security, and reliable deployments for your AI Chatbot Web GUI project! 🚀