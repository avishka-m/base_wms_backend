import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from prophet import Prophet
from .prophet_forecaster import ProphetForecaster

class TestAddExternalRegressors:
    """Test cases for ProphetForecaster.add_external_regressors method"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with external regressor columns"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'ds': dates,
            'y': np.random.randint(10, 100, 100),
            'is_weekend': np.random.randint(0, 2, 100),
            'is_holiday': np.random.randint(0, 2, 100),
            'is_month_end': np.random.randint(0, 2, 100),
            'temperature': np.random.uniform(20, 35, 100),
            'promotion': np.random.randint(0, 2, 100)
        })

    @pytest.fixture
    def mock_prophet_model(self):
        """Create a mock Prophet model"""
        model = Mock(spec=Prophet)
        model.add_regressor = Mock()
        return model

    @pytest.fixture
    def forecaster_with_default_config(self):
        """Create forecaster with default configuration"""
        return ProphetForecaster()

    @pytest.fixture
    def forecaster_with_custom_config(self):
        """Create forecaster with custom external features config"""
        config = {
            'external_features': ['temperature', 'promotion', 'is_weekend']
        }
        return ProphetForecaster(model_config=config)

    def test_add_default_external_regressors(self, forecaster_with_default_config, mock_prophet_model, sample_data):
        """Test adding default external regressors when no config is provided"""
        with patch('logging.getLogger'):
            result_model = forecaster_with_default_config.add_external_regressors(
                mock_prophet_model, sample_data
            )
        
        # Should add default regressors that exist in data
        expected_calls = [
            pytest.mock.call('is_holiday'),
            pytest.mock.call('is_weekend'),
            pytest.mock.call('is_month_end')
        ]
        
        assert mock_prophet_model.add_regressor.call_count == 3
        for call in expected_calls:
            assert call in mock_prophet_model.add_regressor.call_args_list
        
        assert result_model is mock_prophet_model

    def test_add_external_regressors_from_model_config(self, forecaster_with_custom_config, mock_prophet_model, sample_data):
        """Test adding external regressors from model config"""
        with patch('logging.getLogger'):
            result_model = forecaster_with_custom_config.add_external_regressors(
                mock_prophet_model, sample_data
            )
        
        # Should add regressors from config that exist in data
        expected_calls = [
            pytest.mock.call('temperature'),
            pytest.mock.call('promotion'),
            pytest.mock.call('is_weekend')
        ]
        
        assert mock_prophet_model.add_regressor.call_count == 3
        for call in expected_calls:
            assert call in mock_prophet_model.add_regressor.call_args_list

    @patch('base_wms_backend.ai_services.seasonal_inventory.src.models.prophet_forecaster.FEATURE_CONFIG', 
           {'external_features': ['temperature', 'is_holiday']})
    def test_add_external_regressors_from_feature_config(self, forecaster_with_default_config, mock_prophet_model, sample_data):
        """Test adding external regressors from FEATURE_CONFIG when model config doesn't have external_features"""
        # Remove external_features from model config to trigger FEATURE_CONFIG fallback
        forecaster_with_default_config.model_config = {}
        
        with patch('logging.getLogger'):
            result_model = forecaster_with_default_config.add_external_regressors(
                mock_prophet_model, sample_data
            )
        
        # Should add regressors from FEATURE_CONFIG
        expected_calls = [
            pytest.mock.call('temperature'),
            pytest.mock.call('is_holiday')
        ]
        
        assert mock_prophet_model.add_regressor.call_count == 2
        for call in expected_calls:
            assert call in mock_prophet_model.add_regressor.call_args_list

    def test_skip_missing_columns(self, forecaster_with_custom_config, mock_prophet_model):
        """Test that missing columns are skipped without error"""
        # Data without some configured regressors
        data_with_missing = pd.DataFrame({
            'ds': pd.date_range('2023-01-01', periods=10),
            'y': range(10),
            'temperature': range(10)
            # Missing 'promotion' and 'is_weekend'
        })
        
        with patch('logging.getLogger'):
            result_model = forecaster_with_custom_config.add_external_regressors(
                mock_prophet_model, data_with_missing
            )
        
        # Should only add 'temperature' since others don't exist
        mock_prophet_model.add_regressor.assert_called_once_with('temperature')

    def test_no_external_regressors_available(self, forecaster_with_default_config, mock_prophet_model):
        """Test behavior when no external regressors are available in data"""
        # Data with only required columns
        minimal_data = pd.DataFrame({
            'ds': pd.date_range('2023-01-01', periods=10),
            'y': range(10)
        })
        
        with patch('logging.getLogger'):
            result_model = forecaster_with_default_config.add_external_regressors(
                mock_prophet_model, minimal_data
            )
        
        # Should not add any regressors
        mock_prophet_model.add_regressor.assert_not_called()
        assert result_model is mock_prophet_model

    def test_empty_external_features_config(self, mock_prophet_model, sample_data):
        """Test with empty external_features in config"""
        config = {'external_features': []}
        forecaster = ProphetForecaster(model_config=config)
        
        with patch('logging.getLogger'):
            result_model = forecaster.add_external_regressors(mock_prophet_model, sample_data)
        
        # Should not add any regressors when config is empty
        mock_prophet_model.add_regressor.assert_not_called()

    def test_logging_behavior(self, forecaster_with_custom_config, mock_prophet_model, sample_data):
        """Test that proper logging occurs when adding regressors"""
        with patch('logging.getLogger') as mock_logger:
            mock_logger_instance = Mock()
            mock_logger.return_value = mock_logger_instance
            
            forecaster_with_custom_config.add_external_regressors(mock_prophet_model, sample_data)
            
            # Should log each added regressor
            expected_log_calls = [
                pytest.mock.call(' Added external regressor: temperature'),
                pytest.mock.call(' Added external regressor: promotion'),
                pytest.mock.call(' Added external regressor: is_weekend')
            ]
            
            assert mock_logger_instance.info.call_count == 3
            for call in expected_log_calls:
                assert call in mock_logger_instance.info.call_args_list

    def test_import_error_fallback(self, mock_prophet_model, sample_data):
        """Test fallback to default when FEATURE_CONFIG import fails"""
        forecaster = ProphetForecaster(model_config={})  # No external_features in config
        
        with patch('logging.getLogger'), \
             patch('base_wms_backend.ai_services.seasonal_inventory.src.models.prophet_forecaster.FEATURE_CONFIG', 
                   side_effect=ImportError("Module not found")):
            
            result_model = forecaster.add_external_regressors(mock_prophet_model, sample_data)
            
            # Should fall back to default regressors
            expected_calls = [
                pytest.mock.call('is_holiday'),
                pytest.mock.call('is_weekend'),
                pytest.mock.call('is_month_end')
            ]
            
            assert mock_prophet_model.add_regressor.call_count == 3
            for call in expected_calls:
                assert call in mock_prophet_model.add_regressor.call_args_list

    def test_model_config_none_attribute(self, mock_prophet_model, sample_data):
        """Test when model_config doesn't have external_features attribute"""
        # Create config object without external_features
        config = type('Config', (), {})()
        forecaster = ProphetForecaster(model_config=config)
        
        with patch('logging.getLogger'):
            result_model = forecaster.add_external_regressors(mock_prophet_model, sample_data)
        
        # Should fall back to default regressors
        expected_calls = [
            pytest.mock.call('is_holiday'),
            pytest.mock.call('is_weekend'),
            pytest.mock.call('is_month_end')
        ]
        
        assert mock_prophet_model.add_regressor.call_count == 3
        for call in expected_calls:
            assert call in mock_prophet_model.add_regressor.call_args_list

    def test_preserves_original_model_reference(self, forecaster_with_default_config, mock_prophet_model, sample_data):
        """Test that the method returns the same model object reference"""
        with patch('logging.getLogger'):
            result_model = forecaster_with_default_config.add_external_regressors(
                mock_prophet_model, sample_data
            )
        
        # Should return the exact same model object
        assert result_model is mock_prophet_model
        assert id(result_model) == id(mock_prophet_model)

    def test_case_sensitivity_of_column_names(self, forecaster_with_default_config, mock_prophet_model):
        """Test that column name matching is case sensitive"""
        # Data with differently cased column names
        case_sensitive_data = pd.DataFrame({
            'ds': pd.date_range('2023-01-01', periods=10),
            'y': range(10),
            'Is_Weekend': range(10),  # Different case
            'IS_HOLIDAY': range(10),  # Different case
            'is_month_end': range(10)  # Correct case
        })
        
        with patch('logging.getLogger'):
            forecaster_with_default_config.add_external_regressors(
                mock_prophet_model, case_sensitive_data
            )
        
        # Should only add the correctly cased column
        mock_prophet_model.add_regressor.assert_called_once_with('is_month_end')

    def test_with_duplicate_regressor_names(self, mock_prophet_model, sample_data):
        """Test behavior when external_features list has duplicates"""
        config = {
            'external_features': ['is_weekend', 'is_holiday', 'is_weekend', 'temperature']
        }
        forecaster = ProphetForecaster(model_config=config)
        
        with patch('logging.getLogger'):
            forecaster.add_external_regressors(mock_prophet_model, sample_data)
        
        # Should add each regressor once, even if listed multiple times
        expected_calls = [
            pytest.mock.call('is_weekend'),
            pytest.mock.call('is_holiday'),
            pytest.mock.call('is_weekend'),  # Will be called again due to duplicate
            pytest.mock.call('temperature')
        ]
        
        assert mock_prophet_model.add_regressor.call_count == 4
        for call in expected_calls:
            assert call in mock_prophet_model.add_regressor.call_args_list