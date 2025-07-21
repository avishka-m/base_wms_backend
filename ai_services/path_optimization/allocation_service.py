# ai_services/path_optimization/allocation_service.py

from typing import Dict, List, Any, Optional
from .warehouse_mapper import warehouse_mapper
from .location_predictor import location_predictor

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
        
        # Map rack groups to location prefixes - SEASONAL OPTIMIZATION ORDER
        # ✨ Higher numbers first for B and P (B07, P07 closest to packing)
        # ✨ Lower numbers first for D (D08, D01 closest to packing due to horizontal distance)
        rack_group_prefixes = {
            'B Rack 1': ['B07', 'B06', 'B05', 'B04', 'B03', 'B02', 'B01'],  # B07 closest to packing
            'B Rack 2': ['B14', 'B13', 'B12', 'B11', 'B10', 'B09', 'B08'],  # B14 closest to packing
            'B Rack 3': ['B21', 'B20', 'B19', 'B18', 'B17', 'B16', 'B15'],  # B21 closest to packing
            'P Rack 1': ['P07', 'P06', 'P05', 'P04', 'P03', 'P02', 'P01'],  # P07 closest to packing
            'P Rack 2': ['P14', 'P13', 'P12', 'P11', 'P10', 'P09', 'P08'],  # P14 closest to packing
            'D Rack 1': ['D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07'],  # D01 closest to packing (x=3)
            'D Rack 2': ['D08', 'D09', 'D10', 'D11', 'D12', 'D13', 'D14']   # D08 closest to packing (x=3)
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
                               preference: str = 'seasonal_optimized') -> str:
        """Select the best location from available locations with seasonal optimization"""
        
        if not available_locations:
            return None
        
        if preference == 'seasonal_optimized':
            # ✨ SEASONAL OPTIMIZATION: Prioritize locations closest to packing point (0,11)
            # Higher numbered slots (B07, P07, D14) are now physically closest to packing
            def seasonal_priority(location_id):
                slot_code = location_id.split('.')[0]  # Get B01 from B01.1
                floor = int(location_id.split('.')[1])  # Get 1 from B01.1
                
                # Get slot coordinates
                coords = self.get_slot_coordinates(slot_code)
                if coords:
                    x, y = coords['x'], coords['y']
                    # Calculate distance from packing point (0,11) instead of receiving (0,0)
                    distance = ((x - 0) ** 2 + (y - 11) ** 2) ** 0.5
                    # Prefer lower floors (multiply floor by small factor)
                    return distance + (floor * 0.1)
                return float('inf')
            
            available_locations.sort(key=seasonal_priority)
            return available_locations[0]
        
        elif preference == 'closest_to_receiving':
            # Original logic for compatibility: Sort by distance from receiving point (0,0)
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
        """Get map coordinates for a slot code with SEASONAL OPTIMIZATION"""
        
        # Map slot codes to coordinates based on seasonal storage optimization
        # ✨ SEASONAL OPTIMIZATION: Higher numbers closer to packing point (0,11)
        slot_coordinates = {}
        
        # B slots (Medium/Bin) - SEASONAL OPTIMIZATION MAPPING
        # B07 at y=8, B06 at y=7, B05 at y=6, B04 at y=5, B03 at y=4, B02 at y=3, B01 at y=2
        for i in range(1, 8):
            slot_coordinates[f"B{str(i).zfill(2)}"] = {'x': 1, 'y': 2 + (i - 1)}  # B01=y2, B02=y3, ..., B07=y8
        
        # B14 at y=8, B13 at y=7, B12 at y=6, B11 at y=5, B10 at y=4, B09 at y=3, B08 at y=2  
        for i in range(8, 15):
            slot_coordinates[f"B{str(i).zfill(2)}"] = {'x': 3, 'y': 2 + (i - 8)}  # B08=y2, B09=y3, ..., B14=y8
        
        # B21 at y=8, B20 at y=7, B19 at y=6, B18 at y=5, B17 at y=4, B16 at y=3, B15 at y=2
        for i in range(15, 22):
            slot_coordinates[f"B{str(i).zfill(2)}"] = {'x': 5, 'y': 2 + (i - 15)}  # B15=y2, B16=y3, ..., B21=y8
        
        # P slots (Small/Pellet) - SEASONAL OPTIMIZATION MAPPING
        # P07 at y=8, P06 at y=7, P05 at y=6, P04 at y=5, P03 at y=4, P02 at y=3, P01 at y=2
        for i in range(1, 8):
            slot_coordinates[f"P{str(i).zfill(2)}"] = {'x': 7, 'y': 2 + (i - 1)}  # P01=y2, P02=y3, ..., P07=y8
        
        # P14 at y=8, P13 at y=7, P12 at y=6, P11 at y=5, P10 at y=4, P09 at y=3, P08 at y=2
        for i in range(8, 15):
            slot_coordinates[f"P{str(i).zfill(2)}"] = {'x': 9, 'y': 2 + (i - 8)}  # P08=y2, P09=y3, ..., P14=y8
        
        # D slots (Large) - SEASONAL OPTIMIZATION MAPPING
        # D07 at x=9, D06 at x=8, D05 at x=7, D04 at x=6, D03 at x=5, D02 at x=4, D01 at x=3
        for i in range(1, 8):
            slot_coordinates[f"D{str(i).zfill(2)}"] = {'x': 3 + (i - 1), 'y': 10}  # D01=x3, D02=x4, ..., D07=x9
        
        # D14 at x=9, D13 at x=8, D12 at x=7, D11 at x=6, D10 at x=5, D09 at x=4, D08 at x=3  
        for i in range(8, 15):
            slot_coordinates[f"D{str(i).zfill(2)}"] = {'x': 3 + (i - 8), 'y': 11}  # D08=x3, D09=x4, ..., D14=x9
        
        return slot_coordinates.get(slot_code, {'x': 0, 'y': 0})
    
    async def allocate_location_for_item(self, 
                                       item_id: int,
                                       category: str,
                                       item_size: str,
                                       quantity: int,
                                       db_collection_seasonal,
                                       db_collection_storage,
                                       db_collection_location_inventory,
                                       preference: str = 'seasonal_optimized') -> Dict[str, Any]:
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