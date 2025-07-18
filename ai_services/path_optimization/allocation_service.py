# ai_services/path_optimization/allocation_service.py

from typing import Dict, List, Any, Optional
from warehouse_mapper import warehouse_mapper
from location_predictor import location_predictor

class LocationAllocationService:
    """Handles allocation of specific storage locations based on predictions"""
    
    def __init__(self):
        self.mapper = warehouse_mapper
        self.predictor = location_predictor
    
    async def get_available_locations_from_inventory(self, db_collection_location_inventory, location_type: str = None) -> List[str]:
        """Get list of all available locations from location_inventory collection"""
        try:
            # Build query for available locations
            query = {"available": True}
            if location_type:
                query["type"] = location_type.upper()
            
            # Query location inventory for available locations
            available_docs = db_collection_location_inventory.find(query)
            
            available_locations = []
            for doc in available_docs:
                location_id = doc.get('locationID')
                if location_id:
                    available_locations.append(location_id)
            
            print(f"✅ Found {len(available_locations)} available locations of type {location_type or 'ALL'}")
            return available_locations
            
        except Exception as e:
            print(f"❌ Error getting available locations: {str(e)}")
            return []
    
    def filter_locations_by_rack_group(self, available_locations: List[str], target_rack_group: str) -> List[str]:
        """Filter available locations to only include those in the target rack group"""
        
        # Map rack groups to location prefixes
        rack_group_prefixes = {
            'B Rack 1': ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07'],
            'B Rack 2': ['B08', 'B09', 'B10', 'B11', 'B12', 'B13', 'B14'], 
            'B Rack 3': ['B15', 'B16', 'B17', 'B18', 'B19', 'B20', 'B21'],
            'P Rack 1': ['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07'],
            'P Rack 2': ['P08', 'P09', 'P10', 'P11', 'P12', 'P13', 'P14'],
            'D Rack 1': ['D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07'],
            'D Rack 2': ['D08', 'D09', 'D10', 'D11', 'D12', 'D13', 'D14']
        }
        
        prefixes = rack_group_prefixes.get(target_rack_group, [])
        if not prefixes:
            print(f"⚠️ Unknown rack group: {target_rack_group}")
            return []
        
        # Filter locations that start with any of the target prefixes
        filtered_locations = []
        for location in available_locations:
            slot_code = location.split('.')[0]  # Get B01 from B01.1
            if slot_code in prefixes:
                filtered_locations.append(location)
        
        print(f"✅ Filtered to {len(filtered_locations)} locations in {target_rack_group}")
        return filtered_locations
    
    def get_fallback_rack_sequence(self, predicted_rack_group: str) -> List[str]:
        """Get fallback sequence for rack groups of the same type"""
        
        fallback_sequences = {
            # B Rack fallbacks
            'B Rack 1': ['B Rack 1', 'B Rack 2', 'B Rack 3'],
            'B Rack 2': ['B Rack 2', 'B Rack 1', 'B Rack 3'],
            'B Rack 3': ['B Rack 3', 'B Rack 1', 'B Rack 2'],
            
            # P Rack fallbacks
            'P Rack 1': ['P Rack 1', 'P Rack 2'],
            'P Rack 2': ['P Rack 2', 'P Rack 1'],
            
            # D Rack fallbacks
            'D Rack 1': ['D Rack 1', 'D Rack 2'],
            'D Rack 2': ['D Rack 2', 'D Rack 1']
        }
        
        return fallback_sequences.get(predicted_rack_group, [predicted_rack_group])
    
    def select_optimal_location(self, 
                               available_locations: List[str], 
                               preference: str = 'closest_to_receiving') -> str:
        """Select the best location from available locations"""
        
        if not available_locations:
            return None
        
        if preference == 'closest_to_receiving':
            # Sort by distance from receiving point (0,0) and prefer lower floors
            def location_priority(location_id):
                slot_code = location_id.split('.')[0]  # Get B01 from B01.1
                floor = int(location_id.split('.')[1])  # Get 1 from B01.1
                
                # Get slot coordinates
                coords = self.get_slot_coordinates(slot_code)
                if coords:
                    x, y = coords['x'], coords['y']
                    distance = (x ** 2 + y ** 2) ** 0.5  # Euclidean distance from (0,0)
                    # Prefer lower floors (multiply floor by small factor)
                    return distance + (floor * 0.1)
                return float('inf')
            
            available_locations.sort(key=location_priority)
            return available_locations[0]
        
        elif preference == 'lowest_floor_first':
            # Prefer lower floors first, then closest to receiving
            def floor_priority(location_id):
                floor = int(location_id.split('.')[1])
                slot_code = location_id.split('.')[0]
                coords = self.get_slot_coordinates(slot_code)
                distance = 0
                if coords:
                    x, y = coords['x'], coords['y']
                    distance = (x ** 2 + y ** 2) ** 0.5
                return (floor, distance)  # Sort by floor first, then distance
            
            available_locations.sort(key=floor_priority)
            return available_locations[0]
        
        else:
            # Default: return first available
            return available_locations[0]
    
    def get_slot_coordinates(self, slot_code: str) -> Dict[str, int]:
        """Get map coordinates for a slot code (e.g., B01 -> {x: 1, y: 2})"""
        
        # Map slot codes to coordinates based on your layout
        slot_coordinates = {}
        
        # B slots (Medium/Bin)
        # B01-B07: column 1 (x=1), rows 2-8
        for i in range(1, 8):
            slot_coordinates[f"B{str(i).zfill(2)}"] = {'x': 1, 'y': 1 + i}
        
        # B08-B14: column 2 (x=3), rows 2-8  
        for i in range(8, 15):
            slot_coordinates[f"B{str(i).zfill(2)}"] = {'x': 3, 'y': i - 6}
        
        # B15-B21: column 3 (x=5), rows 2-8
        for i in range(15, 22):
            slot_coordinates[f"B{str(i).zfill(2)}"] = {'x': 5, 'y': i - 13}
        
        # P slots (Small/Pellet)  
        # P01-P07: column 1 (x=7), rows 2-8
        for i in range(1, 8):
            slot_coordinates[f"P{str(i).zfill(2)}"] = {'x': 7, 'y': 1 + i}
        
        # P08-P14: column 2 (x=9), rows 2-8
        for i in range(8, 15):
            slot_coordinates[f"P{str(i).zfill(2)}"] = {'x': 9, 'y': i - 6}
        
        # D slots (Large)
        # D01-D07: row 1 (y=10), columns 3-9
        for i in range(1, 8):
            slot_coordinates[f"D{str(i).zfill(2)}"] = {'x': 2 + i, 'y': 10}
        
        # D08-D14: row 2 (y=11), columns 3-9  
        for i in range(8, 15):
            slot_coordinates[f"D{str(i).zfill(2)}"] = {'x': i - 5, 'y': 11}
        
        return slot_coordinates.get(slot_code, {'x': 0, 'y': 0})
    
    async def allocate_location_for_item(self, 
                                       item_id: int,
                                       category: str,
                                       item_size: str,
                                       quantity: int,
                                       db_collection_seasonal,
                                       db_collection_storage,
                                       db_collection_location_inventory,
                                       preference: str = 'closest_to_receiving') -> Dict[str, Any]:
        """Main method to allocate a specific location for an item using improved fallback logic"""
        
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
            print(f"🤖 ML predicted rack group: {predicted_rack_group}")
            
            # Step 2: Get available locations from location inventory  
            location_type = self._get_location_type_from_rack_group(predicted_rack_group)
            available_locations = await self.get_available_locations_from_inventory(
                db_collection_location_inventory, 
                location_type
            )
            
            if not available_locations:
                return {
                    'success': False,
                    'error': f'No available locations of type {location_type} in entire warehouse',
                    'allocated_location': None,
                    'predicted_rack_group': predicted_rack_group,
                    'allocation_reason': f'Warehouse has no {location_type} type locations available'
                }
            
            # ✨ IMPROVED: Step 3 - Try fallback sequence for same rack type
            fallback_sequence = self.get_fallback_rack_sequence(predicted_rack_group)
            print(f"🔄 Trying rack sequence: {fallback_sequence}")
            
            selected_location = None
            used_rack_group = None
            allocation_reason = None
            
            for rack_group in fallback_sequence:
                # Filter to current rack group
                rack_locations = self.filter_locations_by_rack_group(available_locations, rack_group)
                
                if rack_locations:
                    # Found available locations in this rack group
                    selected_location = self.select_optimal_location(rack_locations, preference)
                    used_rack_group = rack_group
                    
                    if rack_group == predicted_rack_group:
                        allocation_reason = f"ML model predicted {predicted_rack_group} with {prediction_result['confidence']:.2f} confidence"
                    else:
                        allocation_reason = f"ML predicted {predicted_rack_group} (full) → Using fallback {rack_group}"
                    
                    print(f"✅ Found location in {rack_group}: {selected_location}")
                    break
                else:
                    print(f"❌ No available locations in {rack_group}")
            
            # ✨ IMPROVED: If no locations found in any rack of the same type
            if not selected_location:
                rack_type_names = {
                    'M': 'B Racks (Medium/Bin)',
                    'S': 'P Racks (Small/Pellet)', 
                    'D': 'D Racks (Large)'
                }
                
                return {
                    'success': False,
                    'error': f'No available space to store in {rack_type_names.get(location_type, "any racks")}',
                    'allocated_location': None,
                    'predicted_rack_group': predicted_rack_group,
                    'checked_racks': fallback_sequence,
                    'allocation_reason': f'All {rack_type_names.get(location_type)} are full'
                }
            
            # Step 4: Get coordinates for the selected location
            slot_code = selected_location.split('.')[0]
            floor = int(selected_location.split('.')[1])
            coordinates = self.get_slot_coordinates(slot_code)
            
            return {
                'success': True,
                'allocated_location': selected_location,
                'slot_code': slot_code,
                'coordinates': {
                    'x': coordinates.get('x', 0),
                    'y': coordinates.get('y', 0),
                    'floor': floor
                },
                'predicted_rack_group': predicted_rack_group,
                'used_rack_group': used_rack_group,
                'confidence': prediction_result['confidence'],
                'total_available_locations': len(available_locations),
                'checked_racks': fallback_sequence,
                'allocation_reason': allocation_reason
            }
            
        except Exception as e:
            print(f"❌ Error in location allocation: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'allocated_location': None
            }
    
    def _get_location_type_from_rack_group(self, rack_group: str) -> str:
        """Convert rack group to location type"""
        if 'B Rack' in rack_group:
            return 'M'  # Medium
        elif 'P Rack' in rack_group:
            return 'S'  # Small
        elif 'D Rack' in rack_group:
            return 'D'  # Large
        return 'M'  # Default to Medium
    
    def _get_rack_group_from_location(self, location_id: str) -> str:
        """Get rack group from location ID"""
        slot_code = location_id.split('.')[0]
        
        if slot_code.startswith('B'):
            slot_num = int(slot_code[1:])
            if 1 <= slot_num <= 7:
                return 'B Rack 1'
            elif 8 <= slot_num <= 14:
                return 'B Rack 2'
            elif 15 <= slot_num <= 21:
                return 'B Rack 3'
        elif slot_code.startswith('P'):
            slot_num = int(slot_code[1:])
            if 1 <= slot_num <= 7:
                return 'P Rack 1'
            elif 8 <= slot_num <= 14:
                return 'P Rack 2'
        elif slot_code.startswith('D'):
            slot_num = int(slot_code[1:])
            if 1 <= slot_num <= 7:
                return 'D Rack 1'
            elif 8 <= slot_num <= 14:
                return 'D Rack 2'
        
        return 'Unknown'
    
    def get_location_details(self, location_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific location"""
        
        slot_code = location_id.split('.')[0]
        floor = int(location_id.split('.')[1]) if '.' in location_id else 1
        coordinates = self.get_slot_coordinates(slot_code)
        rack_group = self._get_rack_group_from_location(location_id)
        
        return {
            'valid': True,
            'location_id': location_id,
            'slot_code': slot_code,
            'floor': floor,
            'coordinates': coordinates,
            'rack_group': rack_group
        }

# Global instance
allocation_service = LocationAllocationService()