<!-- # Warehouse Management System Backend

This is the backend API for the Warehouse Management System (WMS), a comprehensive solution for inventory tracking, order processing, and logistics optimization.

## Features

- **Inventory Management**: Track stock levels, item details, and storage locations
- **Order Processing**: Manage the complete order lifecycle from creation to delivery
- **Warehouse Operations**: Handle receiving, picking, packing, shipping, and returns
- **Role-Based Access Control**: Secure API with different permission levels for managers, workers, and customers
- **Vehicle Management**: Track vehicle availability and maintenance
- **Analytics & Reporting**: Monitor warehouse utilization and detect inventory anomalies

## Tech Stack

- **Framework**: FastAPI
- **Database**: MongoDB
- **Authentication**: JWT (JSON Web Tokens)
- **Async Support**: Full support for asynchronous operations
- **Typing**: Type annotations throughout the codebase for better IDE support and type safety

## Getting Started

### Prerequisites

- Python 3.10+
- MongoDB 5.0+ -->

### Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file with the following variables:
   ```
   MONGODB_URL=mongodb://localhost:27017
   DATABASE_NAME=warehouse_management
   SECRET_KEY=your-secret-key-here
   DEBUG_MODE=True
   ```

### Running the Application

Start the development server with:

```
uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000

Interactive API documentation is automatically generated and available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Structure

The API is organized into the following modules:

- `/api/v1/auth`: Authentication endpoints
- `/api/v1/inventory`: Inventory management
- `/api/v1/orders`: Order processing
- `/api/v1/workers`: Worker management
- `/api/v1/customers`: Customer management
- `/api/v1/locations`: Warehouse and storage location management
- `/api/v1/receiving`: Handling incoming inventory
- `/api/v1/picking`: Order item picking
- `/api/v1/packing`: Order packing
- `/api/v1/shipping`: Shipping and delivery
- `/api/v1/returns`: Returns processing
- `/api/v1/vehicles`: Vehicle management

## Authentication

The API uses JWT-based authentication. To access protected endpoints:

1. Obtain a token using the `/api/v1/auth/token` endpoint
2. Include the token in an Authorization header: `Authorization: Bearer {token}`

## Role-Based Access

The API supports the following roles with different permission levels:

- **Manager**: Full access to all endpoints
- **ReceivingClerk**: Access to receiving and returns endpoints
- **Picker**: Access to picking endpoints
- **Packer**: Access to packing endpoints
- **Driver**: Access to shipping and vehicle endpoints
- **Customer**: Limited access to their own orders and returns

## Data Models

The system uses the following main data models:

- **Inventory**: Items in stock
- **Order**: Customer orders
- **Worker**: Staff members with different roles
- **Customer**: People who place orders
- **Location**: Storage locations in the warehouse
- **Warehouse**: Physical warehouse facilities
- **Vehicle**: Delivery vehicles

## Contributing

1. Create a feature branch
2. Implement your feature or bug fix
3. Write or update tests
4. Submit a pull request

## License

This project is licensed under the MIT License.