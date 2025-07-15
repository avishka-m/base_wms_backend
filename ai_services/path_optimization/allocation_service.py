# ai-services/path_optimization/allocation_service.py

from typing import Dict, List, Any, Optional
from warehouse_mapper import warehouse_mapper
from location_predictor import location_predictor

class LocationAllocationService:
    """Handles allocation of specific storage locations based on predictions"""
    
    def __init__(self):
        self.mapper = warehouse_mapper
        self.predictor = location_predictor
    
    async def get_occupied_sublocations(self, db_collection_storage) -> List[str]:
        """Get list of all currently occupied sublocations"""
        try:
            # Query storage history for occupied locations
            occupied_docs = db_collection_storage.find({
                "action": "stored",
                # Add any additional filters for active storage
            })
            
            occupied_sublocations = []
            for doc in occupied_docs:
                location_id = doc.get('locationID')
                if location_id:
                    occupied_sublocations.append(location_id)
            
            return occupied_sublocations
            
        except Exception as e:
            print(f"Error getting occupied locations: {str(e)}")
            return []
    
    def get_free_sublocations_in_rack_group(self, 
                                          rack_group: str, 
                                          occupied_sublocations: List[str]) -> List[str]:
        """Get all free sublocations in a specific rack group"""
        
        # Get all base locations in the rack group
        base_locations = self.mapper.get_rack_locations(rack_group)
        
        free_sublocations = []
        
        for base_location in base_locations:
            # Get all 4 sublocations for this base location
            all_sublocations = self.mapper.get_all_sublocations_for_base(base_location)
            
            # Filter out occupied ones
            for sublocation in all_sublocations:
                if sublocation not in occupied_sublocations:
                    free_sublocations.append(sublocation)
        
        return free_sublocations
    
    def select_optimal_sublocation(self, 
                                 free_sublocations: List[str], 
                                 preference: str = 'closest_to_receiving') -> str:
        """Select the best sublocation from available free locations"""
        
        if not free_sublocations:
            return None
        
        if preference == 'closest_to_receiving':
            # Sort by distance from receiving point (0,0)
            def distance_from_receiving(sublocation):
                coords = self.mapper.get_coordinates(sublocation)
                if coords:
                    x, y = coords.get('x', 0), coords.get('y', 0)
                    return (x ** 2 + y ** 2) ** 0.5  # Euclidean distance
                return float('inf')
            
            free_sublocations.sort(key=distance_from_receiving)
            return free_sublocations[0]
        
        elif preference == 'lowest_floor_first':
            # Prefer lower floors (easier access)
            def floor_priority(sublocation):
                if '.' in sublocation:
                    floor = int(sublocation.split('.')[1])
                    return floor
                return 1
            
            free_sublocations.sort(key=floor_priority)
            return free_sublocations[0]
        
        else:
            # Default: return first available
            return free_sublocations[0]
    
    async def allocate_location_for_item(self, 
                                       item_id: int,
                                       category: str,
                                       item_size: str,
                                       quantity: int,
                                       db_collection_seasonal,
                                       db_collection_storage,
                                       preference: str = 'closest_to_receiving') -> Dict[str, Any]:
        """Main method to allocate a specific location for an item"""
        
        try:
            # Step 1: Predict optimal rack group using ML model
            prediction_result = await self.predictor.predict_optimal_location(
                item_id=item_id,
                category=category,
                item_size=item_size,
                db_collection_seasonal=db_collection_seasonal,
                quantity=quantity
            )
            
            predicted_rack_group = prediction_result['rack_group']
            
            # Step 2: Get currently occupied sublocations
            occupied_sublocations = await self.get_occupied_sublocations(db_collection_storage)
            
            # Step 3: Find free sublocations in the predicted rack group
            free_sublocations = self.get_free_sublocations_in_rack_group(
                predicted_rack_group, 
                occupied_sublocations
            )
            
            # Step 4: Select optimal sublocation
            selected_sublocation = self.select_optimal_sublocation(
                free_sublocations, 
                preference
            )
            
            if not selected_sublocation:
                # Fallback: try other rack groups of same type
                return await self._fallback_allocation(
                    item_size, 
                    occupied_sublocations, 
                    preference,
                    prediction_result
                )
            
            # Step 5: Get coordinates for the selected location
            coordinates = self.mapper.get_coordinates(selected_sublocation)
            
            return {
                'success': True,
                'allocated_location': selected_sublocation,
                'base_location': self.mapper.parse_sublocation_to_base(selected_sublocation),
                'coordinates': {
                    'x': coordinates.get('x', 0),
                    'y': coordinates.get('y', 0),
                    'floor': int(selected_sublocation.split('.')[1]) if '.' in selected_sublocation else 1
                },
                'predicted_rack_group': predicted_rack_group,
                'confidence': prediction_result['confidence'],
                'total_free_locations': len(free_sublocations),
                'allocation_reason': f"ML model predicted {predicted_rack_group} with {prediction_result['confidence']:.2f} confidence"
            }
            
        except Exception as e:
            print(f"Error in location allocation: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'allocated_location': None
            }
    
    async def _fallback_allocation(self, 
                                 item_size: str, 
                                 occupied_sublocations: List[str],
                                 preference: str,
                                 prediction_result: Dict) -> Dict[str, Any]:
        """Fallback allocation when predicted rack group is full"""
        
        # Define fallback rack groups based on item size
        fallback_groups = []
        
        if item_size.upper() == 'S':
            fallback_groups = ['P Rack 1', 'P Rack 2', 'B Rack 1', 'B Rack 2', 'B Rack 3']
        elif item_size.upper() == 'L':
            fallback_groups = ['D Rack 1', 'D Rack 2', 'B Rack 3', 'B Rack 2', 'B Rack 1']
        else:  # Medium
            fallback_groups = ['B Rack 1', 'B Rack 2', 'B Rack 3', 'P Rack 1', 'P Rack 2']
        
        for rack_group in fallback_groups:
            free_sublocations = self.get_free_sublocations_in_rack_group(
                rack_group, 
                occupied_sublocations
            )
            
            if free_sublocations:
                selected_sublocation = self.select_optimal_sublocation(
                    free_sublocations, 
                    preference
                )
                
                coordinates = self.mapper.get_coordinates(selected_sublocation)
                
                return {
                    'success': True,
                    'allocated_location': selected_sublocation,
                    'base_location': self.mapper.parse_sublocation_to_base(selected_sublocation),
                    'coordinates': {
                        'x': coordinates.get('x', 0),
                        'y': coordinates.get('y', 0),
                        'floor': int(selected_sublocation.split('.')[1]) if '.' in selected_sublocation else 1
                    },
                    'predicted_rack_group': prediction_result['rack_group'],
                    'actual_rack_group': rack_group,
                    'confidence': prediction_result['confidence'],
                    'total_free_locations': len(free_sublocations),
                    'allocation_reason': f"Fallback allocation - predicted {prediction_result['rack_group']} was full"
                }
        
        # If no space found anywhere
        return {
            'success': False,
            'error': 'No free locations available in warehouse',
            'allocated_location': None,
            'predicted_rack_group': prediction_result['rack_group'],
            'allocation_reason': 'Warehouse is full'
        }
    
    def get_location_details(self, sublocation: str) -> Dict[str, Any]:
        """Get detailed information about a specific location"""
        
        if not self.mapper.validate_location(sublocation):
            return {'valid': False, 'error': 'Invalid location code'}
        
        base_location = self.mapper.parse_sublocation_to_base(sublocation)
        coordinates = self.mapper.get_coordinates(sublocation)
        rack_group = self.mapper.get_rack_group_from_location(sublocation)
        
        return {
            'valid': True,
            'sublocation': sublocation,
            'base_location': base_location,
            'coordinates': coordinates,
            'rack_group': rack_group,
            'floor': int(sublocation.split('.')[1]) if '.' in sublocation else 1
        }

# Global instance
allocation_service = LocationAllocationService()