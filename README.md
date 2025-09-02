# WMS Backend API

A comprehensive Warehouse Management System (WMS) backend built with FastAPI, MongoDB, and modern Python practices.

## 🚀 Features

- **User Authentication & Authorization** - JWT-based auth with role-based access control
- **Inventory Management** - Track items, locations, and stock levels
- **Order Processing** - Complete order lifecycle from creation to fulfillment
- **Picking & Packing** - Automated picking job creation and workflow management
- **Returns Processing** - Handle item returns and approval workflows
- **Location Management** - Multi-floor warehouse location tracking
- **AI-Powered Analytics** - Inventory optimization, storage location prediction,agentic chatbot, stock anomaly detection, and seasonal demand forecasting
- **Real-time Updates** - WebSocket support for live data updates
- **RESTful API** - Comprehensive REST endpoints with automatic documentation

## 🛠️ Tech Stack

- **Framework**: FastAPI
- **Database**: MongoDB
- **Authentication**: JWT tokens with bcrypt password hashing
- **Documentation**: Auto-generated OpenAPI/Swagger docs
- **Data Validation**: Pydantic models
- **AI/ML**: Prophet forecasting, scikit-learn
- **Testing**: pytest
- **CORS**: Configured for frontend integration

## 📋 Prerequisites

- Python 3.8+
- MongoDB (local or Atlas)
- Git

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/base_wms_backend.git
cd base_wms_backend
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Database

Create a `.env` file in the root directory:

```env
# MongoDB Configuration
MONGODB_URL=mongodb://localhost:27017/
DATABASE_NAME=warehouse_management

# For MongoDB Atlas (optional):
# ATLAS_USERNAME=your_username
# ATLAS_PASSWORD=your_password
# ATLAS_CLUSTER_HOST=your_cluster.mongodb.net

# JWT Configuration
SECRET_KEY=your-super-secret-jwt-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=60

# API Configuration
API_V1_PREFIX=/api/v1
DEBUG_MODE=true
```

### 4. Initialize Database & Users

```bash
# Create basic users (manager, receiver, picker, driver, packer)
python create_basic_users.py

# Initialize warehouse locations
# (requires authentication - see API Usage section)
```

### 5. Run the Server

```bash
# Development server
python run.py

# Or with uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

The API will be available at:
- **API**: http://localhost:8002
- **Swagger Docs**: http://localhost:8002/docs
- **ReDoc**: http://localhost:8002/redoc

## 👥 Default Users

After running `create_basic_users.py`, you'll have these accounts:

| Username | Password | Role | Permissions |
|----------|----------|------|-------------|
| `manager` | `manager123` | Manager | All operations |
| `receiver` | `receiver123` | ReceivingClerk | Inventory, Receiving |
| `picker` | `picker123` | Picker | Picking operations |
| `driver` | `driver123` | Driver | Shipping operations |
| `packer` | `packer123` | Packer | Packing operations |

## 🔐 API Usage

### Authentication

1. **Login to get JWT token:**
```bash
curl -X POST "http://localhost:8002/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=manager&password=manager123"
```

2. **Use token in requests:**
```bash
curl -X GET "http://localhost:8002/api/v1/orders" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN_HERE"
```

### Key Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| `POST` | `/api/v1/auth/login` | User login | No |
| `GET` | `/api/v1/orders` | List orders | Yes |
| `POST` | `/api/v1/orders` | Create order | Yes |
| `GET` | `/api/v1/inventory` | List inventory | Yes |
| `POST` | `/api/v1/picking/jobs` | Create picking job | Yes |
| `GET` | `/api/v1/locations` | List locations | Yes |
| `POST` | `/api/v1/location-inventory/initialize-locations` | Initialize warehouse | Yes |

## 📊 Database Collections

- **users** - User accounts and authentication
- **customers** - Customer information
- **inventory** - Product catalog and stock levels
- **location_inventory** - Item locations in warehouse
- **locations** - Warehouse location definitions
- **orders** - Customer orders and order items
- **picking** - Picking jobs and tasks
- **returns** - Return requests and processing
- **workers** - Worker information and assignments

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test files
python test_orders_api_direct.py
python test_picking_simple.py
python test_inventory_total_stock.py

# Test order creation workflow
python test_simple_order.py
```

## 📁 Project Structure

```
base_wms_backend/
├── app/
│   ├── api/              # API endpoints
│   ├── auth/             # Authentication & authorization
│   ├── core/             # Core business logic
│   ├── models/           # Pydantic models
│   ├── services/         # Business logic services
│   ├── tools/            # Utility tools and helpers
│   ├── utils/            # Common utilities
│   ├── config.py         # Configuration settings
│   └── main.py           # FastAPI application
├── ai_services/          # AI/ML services
├── tests/                # Test files
├── requirements.txt      # Python dependencies
├── run.py               # Server startup script
└── README.md            # This file
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MONGODB_URL` | MongoDB connection string | `mongodb://localhost:27017/` |
| `DATABASE_NAME` | Database name | `warehouse_management` |
| `SECRET_KEY` | JWT secret key | Required |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Token expiry | `60` |
| `API_V1_PREFIX` | API prefix | `/api/v1` |
| `DEBUG_MODE` | Debug mode | `false` |

### MongoDB Atlas Setup

For production, configure MongoDB Atlas:

1. Create cluster and get connection details
2. Set environment variables:
```env
ATLAS_USERNAME=your_username
ATLAS_PASSWORD=your_password
ATLAS_CLUSTER_HOST=your-cluster.mongodb.net
```

## 🚀 Deployment

### Docker (Coming Soon)

```bash
# Build image
docker build -t wms-backend .

# Run container
docker run -p 8002:8002 wms-backend
```

### Production Considerations

- Use environment variables for all secrets
- Enable HTTPS/TLS
- Configure proper CORS origins
- Set up monitoring and logging
- Use production WSGI server (Gunicorn)
- Implement rate limiting

## 📝 API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8002/docs
- **ReDoc**: http://localhost:8002/redoc

## 🔍 Troubleshooting

### Common Issues

1. **401 Unauthorized**: Make sure you're using a valid JWT token
2. **Connection refused**: Check if MongoDB is running
3. **Module not found**: Ensure virtual environment is activated
4. **Port already in use**: Change port in `run.py` or stop conflicting service

### Debug Mode

Enable debug mode in `.env`:
```env
DEBUG_MODE=true
```

### Database Issues

```bash
# Check MongoDB connection
python -c "from app.utils.database import get_collection; print('Connected:', get_collection('test') is not None)"

# Reset database (careful!)
python cleanup_orders.py
```

## 🎯 Roadmap

- [ ] Docker containerization
- [ ] Automated testing CI/CD
- [ ] Advanced reporting features
- [ ] Mobile app API support
- [ ] Multi-warehouse support
- [ ] Integration with external systems

