# Configuration Structure

## Overview
The configuration system has been reorganized into a cleaner, more maintainable structure that eliminates duplication and provides clear separation of concerns.

## Structure

```
base_wms_backend/
├── config/
│   ├── __init__.py                    # Configuration package
│   └── base.py                        # Shared configuration settings
├── app/
│   └── config.py                      # Main WMS application configuration
└── ai_services/
    └── seasonal_inventory/
        └── config.py                  # AI forecasting configuration
```

## Configuration Files

### 1. Base Configuration (`config/base.py`)
**Purpose**: Shared settings used across multiple modules
**Contains**:
- Database connection settings (MongoDB URL, database name)
- Shared project information
- Environment settings
- Common logging configuration
- Utility functions for configuration validation

### 2. Main App Configuration (`app/config.py`)
**Purpose**: WMS application-specific settings
**Contains**:
- Security settings (SECRET_KEY, tokens)
- Chatbot configuration (OpenAI, roles, development mode)
- API endpoint mappings
- Chroma DB configuration for knowledge base
- WMS-specific features

**Imports from base**: Database settings, logging, environment

### 3. AI Services Configuration (`ai_services/seasonal_inventory/config.py`)
**Purpose**: AI/ML forecasting module settings
**Contains**:
- Prophet model parameters and configuration
- Data processing settings
- Training and validation parameters
- Feature engineering configuration
- Performance and caching settings
- Business logic for inventory management
- Monitoring and alerting configuration

**Imports from base**: Database settings, environment, project version

## Benefits of New Structure

### ✅ **Advantages**:
1. **No Duplication**: Shared settings are defined once in base config
2. **Consistency**: All modules use the same database and environment settings
3. **Maintainability**: Changes to shared settings only need to be made in one place
4. **Modularity**: Each module can still have its own specific configurations
5. **Clear Dependencies**: Import relationships are explicit and documented

### 🗑️ **Removed**:
- Empty chatbot config file that served no purpose
- Duplicate database configuration
- Inconsistent environment variable handling

## Usage Examples

### Import shared settings:
```python
from base_wms_backend.config.base import MONGODB_URL, DATABASE_NAME
```

### Import app-specific settings:
```python
from base_wms_backend.app.config import OPENAI_API_KEY, ROLES
```

### Import AI-specific settings:
```python
from base_wms_backend.ai_services.seasonal_inventory.config import PROPHET_CONFIG, MODEL_CACHE_STRATEGY
```

## Environment Variables
All configuration files still respect environment variables for deployment flexibility:
- `MONGODB_URL`: Database connection string
- `DATABASE_NAME`: Database name
- `ENVIRONMENT`: Runtime environment (development/production/testing)
- `LOG_LEVEL`: Logging level
- `MODEL_CACHE_STRATEGY`: AI model caching strategy

## Migration Notes
- The empty `app/core/chatbot/config.py` file has been removed
- Import paths have been updated to use the new structure
- All existing functionality is preserved
- No breaking changes to actual configuration values
