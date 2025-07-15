# ai-services/path_optimization/warehouse_mapper.py

class WarehouseMapper:
    """Maps warehouse locations to coordinates and handles location conversions"""
    
    def __init__(self):
        # Define the warehouse grid mapping
        self.warehouse_map = {
            # B Rack 1 (B01-B07) - Column 1, Rows 2-8
            'B01': {'x': 1, 'y': 2, 'rack_group': 'B Rack 1'},
            'B02': {'x': 1, 'y': 3, 'rack_group': 'B Rack 1'},
            'B03': {'x': 1, 'y': 4, 'rack_group': 'B Rack 1'},
            'B04': {'x': 1, 'y': 5, 'rack_group': 'B Rack 1'},
            'B05': {'x': 1, 'y': 6, 'rack_group': 'B Rack 1'},
            'B06': {'x': 1, 'y': 7, 'rack_group': 'B Rack 1'},
            'B07': {'x': 1, 'y': 8, 'rack_group': 'B Rack 1'},
            
            # B Rack 2 (B08-B14) - Column 3, Rows 2-8
            'B08': {'x': 3, 'y': 2, 'rack_group': 'B Rack 2'},
            'B09': {'x': 3, 'y': 3, 'rack_group': 'B Rack 2'},
            'B10': {'x': 3, 'y': 4, 'rack_group': 'B Rack 2'},
            'B11': {'x': 3, 'y': 5, 'rack_group': 'B Rack 2'},
            'B12': {'x': 3, 'y': 6, 'rack_group': 'B Rack 2'},
            'B13': {'x': 3, 'y': 7, 'rack_group': 'B Rack 2'},
            'B14': {'x': 3, 'y': 8, 'rack_group': 'B Rack 2'},
            
            # B Rack 3 (B15-B21) - Column 5, Rows 2-8
            'B15': {'x': 5, 'y': 2, 'rack_group': 'B Rack 3'},
            'B16': {'x': 5, 'y': 3, 'rack_group': 'B Rack 3'},
            'B17': {'x': 5, 'y': 4, 'rack_group': 'B Rack 3'},
            'B18': {'x': 5, 'y': 5, 'rack_group': 'B Rack 3'},
            'B19': {'x': 5, 'y': 6, 'rack_group': 'B Rack 3'},
            'B20': {'x': 5, 'y': 7, 'rack_group': 'B Rack 3'},
            'B21': {'x': 5, 'y': 8, 'rack_group': 'B Rack 3'},
            
            # P Rack 1 (P01-P07) - Column 7, Rows 2-8
            'P01': {'x': 7, 'y': 2, 'rack_group': 'P Rack 1'},
            'P02': {'x': 7, 'y': 3, 'rack_group': 'P Rack 1'},
            'P03': {'x': 7, 'y': 4, 'rack_group': 'P Rack 1'},
            'P04': {'x': 7, 'y': 5, 'rack_group': 'P Rack 1'},
            'P05': {'x': 7, 'y': 6, 'rack_group': 'P Rack 1'},
            'P06': {'x': 7, 'y': 7, 'rack_group': 'P Rack 1'},
            'P07': {'x': 7, 'y': 8, 'rack_group': 'P Rack 1'},
            
            # P Rack 2 (P08-P14) - Column 9, Rows 2-8
            'P08': {'x': 9, 'y': 2, 'rack_group': 'P Rack 2'},
            'P09': {'x': 9, 'y': 3, 'rack_group': 'P Rack 2'},
            'P10': {'x': 9, 'y': 4, 'rack_group': 'P Rack 2'},
            'P11': {'x': 9, 'y': 5, 'rack_group': 'P Rack 2'},
            'P12': {'x': 9, 'y': 6, 'rack_group': 'P Rack 2'},
            'P13': {'x': 9, 'y': 7, 'rack_group': 'P Rack 2'},
            'P14': {'x': 9, 'y': 8, 'rack_group': 'P Rack 2'},
            
            # D Rack 1 (D01-D07) - Row 10, Columns 3-9
            'D01': {'x': 3, 'y': 10, 'rack_group': 'D Rack 1'},
            'D02': {'x': 4, 'y': 10, 'rack_group': 'D Rack 1'},
            'D03': {'x': 5, 'y': 10, 'rack_group': 'D Rack 1'},
            'D04': {'x': 6, 'y': 10, 'rack_group': 'D Rack 1'},
            'D05': {'x': 7, 'y': 10, 'rack_group': 'D Rack 1'},
            'D06': {'x': 8, 'y': 10, 'rack_group': 'D Rack 1'},
            'D07': {'x': 9, 'y': 10, 'rack_group': 'D Rack 1'},
            
            # D Rack 2 (D08-D14) - Row 11, Columns 3-9
            'D08': {'x': 3, 'y': 11, 'rack_group': 'D Rack 2'},
            'D09': {'x': 4, 'y': 11, 'rack_group': 'D Rack 2'},
            'D10': {'x': 5, 'y': 11, 'rack_group': 'D Rack 2'},
            'D11': {'x': 6, 'y': 11, 'rack_group': 'D Rack 2'},
            'D12': {'x': 7, 'y': 11, 'rack_group': 'D Rack 2'},
            'D13': {'x': 8, 'y': 11, 'rack_group': 'D Rack 2'},
            'D14': {'x': 9, 'y': 11, 'rack_group': 'D Rack 2'},
        }
        
        # Group mapping for easy lookup
        self.rack_groups = {
            'B Rack 1': ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07'],
            'B Rack 2': ['B08', 'B09', 'B10', 'B11', 'B12', 'B13', 'B14'],
            'B Rack 3': ['B15', 'B16', 'B17', 'B18', 'B19', 'B20', 'B21'],
            'P Rack 1': ['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07'],
            'P Rack 2': ['P08', 'P09', 'P10', 'P11', 'P12', 'P13', 'P14'],
            'D Rack 1': ['D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07'],
            'D Rack 2': ['D08', 'D09', 'D10', 'D11', 'D12', 'D13', 'D14'],
        }
    
    def get_rack_locations(self, rack_group_name):
        """Get all location codes for a rack group"""
        return self.rack_groups.get(rack_group_name, [])
    
    def get_coordinates(self, location_code):
        """Get x,y coordinates for a location code (e.g., 'B01' -> {'x': 1, 'y': 2})"""
        # Extract base location from sublocation (e.g., 'B1.2' -> 'B01')
        if '.' in location_code:
            base_location = self.parse_sublocation_to_base(location_code)
        else:
            base_location = location_code
            
        return self.warehouse_map.get(base_location, {})
    
    def parse_sublocation_to_base(self, sublocation):
        """Convert sublocation format (B1.2) to base location (B01)"""
        if '.' not in sublocation:
            return sublocation
            
        parts = sublocation.split('.')
        if len(parts) != 2:
            return sublocation
            
        location_part = parts[0]  # e.g., 'B1', 'P4', 'D2'
        
        # Extract prefix and number
        prefix = location_part[0]  # 'B', 'P', or 'D'
        number = int(location_part[1:])  # 1, 4, 2, etc.
        
        # Format as base location (B1 -> B01, B10 -> B10)
        return f"{prefix}{number:02d}"
    
    def parse_base_to_sublocation_format(self, base_location, floor=1):
        """Convert base location (B01) to sublocation format (B1.1)"""
        if len(base_location) < 3:
            return f"{base_location}.{floor}"
            
        prefix = base_location[0]  # 'B', 'P', or 'D'
        number = int(base_location[1:])  # 01 -> 1, 14 -> 14
        
        return f"{prefix}{number}.{floor}"
    
    def get_all_sublocations_for_base(self, base_location):
        """Get all 4 sublocations for a base location (e.g., B01 -> [B1.1, B1.2, B1.3, B1.4])"""
        sublocations = []
        for floor in range(1, 5):  # Floors 1-4
            sublocation = self.parse_base_to_sublocation_format(base_location, floor)
            sublocations.append(sublocation)
        return sublocations
    
    def get_rack_group_from_location(self, location_code):
        """Get rack group name from location code"""
        if '.' in location_code:
            base_location = self.parse_sublocation_to_base(location_code)
        else:
            base_location = location_code
            
        location_info = self.warehouse_map.get(base_location, {})
        return location_info.get('rack_group')
    
    def validate_location(self, location_code):
        """Validate if a location code exists"""
        if '.' in location_code:
            base_location = self.parse_sublocation_to_base(location_code)
            parts = location_code.split('.')
            floor = int(parts[1]) if len(parts) == 2 else 1
            return base_location in self.warehouse_map and 1 <= floor <= 4
        else:
            return location_code in self.warehouse_map

# Global instance
warehouse_mapper = WarehouseMapper()