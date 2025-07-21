#!/usr/bin/env python3
"""
Test script to verify seasonal optimization in allocation service
"""

import sys
import os
import asyncio

# Add the ai_services path to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai_services', 'path_optimization'))

from allocation_service import allocation_service

def test_coordinate_mapping():
    """Test that coordinate mapping now follows seasonal optimization"""
    print("🧪 Testing Coordinate Mapping for Seasonal Optimization")
    print("=" * 60)
    
    # Test B slots - should show B07 closer to packing than B01
    b01_coords = allocation_service.get_slot_coordinates('B01')
    b07_coords = allocation_service.get_slot_coordinates('B07')
    
    print(f"B01 coordinates: {b01_coords}")
    print(f"B07 coordinates: {b07_coords}")
    
    # Calculate distances from packing point (0, 11)
    b01_distance = ((b01_coords['x'] - 0) ** 2 + (b01_coords['y'] - 11) ** 2) ** 0.5
    b07_distance = ((b07_coords['x'] - 0) ** 2 + (b07_coords['y'] - 11) ** 2) ** 0.5
    
    print(f"B01 distance from packing point (0,11): {b01_distance:.2f}")
    print(f"B07 distance from packing point (0,11): {b07_distance:.2f}")
    
    if b07_distance < b01_distance:
        print("✅ B07 is closer to packing point than B01 - SEASONAL OPTIMIZATION WORKING!")
    else:
        print("❌ B01 is still closer to packing point than B07 - NEEDS FIX")
    
    print()
    
    # Test P slots
    p01_coords = allocation_service.get_slot_coordinates('P01')
    p07_coords = allocation_service.get_slot_coordinates('P07')
    
    print(f"P01 coordinates: {p01_coords}")
    print(f"P07 coordinates: {p07_coords}")
    
    p01_distance = ((p01_coords['x'] - 0) ** 2 + (p01_coords['y'] - 11) ** 2) ** 0.5
    p07_distance = ((p07_coords['x'] - 0) ** 2 + (p07_coords['y'] - 11) ** 2) ** 0.5
    
    print(f"P01 distance from packing point (0,11): {p01_distance:.2f}")
    print(f"P07 distance from packing point (0,11): {p07_distance:.2f}")
    
    if p07_distance < p01_distance:
        print("✅ P07 is closer to packing point than P01 - SEASONAL OPTIMIZATION WORKING!")
    else:
        print("❌ P01 is still closer to packing point than P07 - NEEDS FIX")
    
    print()
    
    # Test D slots
    d01_coords = allocation_service.get_slot_coordinates('D01')
    d14_coords = allocation_service.get_slot_coordinates('D14')
    
    print(f"D01 coordinates: {d01_coords}")
    print(f"D14 coordinates: {d14_coords}")
    
    d01_distance = ((d01_coords['x'] - 0) ** 2 + (d01_coords['y'] - 11) ** 2) ** 0.5
    d14_distance = ((d14_coords['x'] - 0) ** 2 + (d14_coords['y'] - 11) ** 2) ** 0.5
    
    print(f"D01 distance from packing point (0,11): {d01_distance:.2f}")
    print(f"D14 distance from packing point (0,11): {d14_distance:.2f}")
    
    if d01_distance < d14_distance:
        print("✅ D01 is closer to packing point than D14 - CORRECT! D slots optimize by distance, not slot number")
    else:
        print("❌ D14 is closer to packing point than D01 - Distance calculation may be wrong")

def test_location_selection():
    """Test that location selection prioritizes seasonal optimization"""
    print("\n🧪 Testing Location Selection Priority")
    print("=" * 60)
    
    # Test with mock available locations for B Rack 1
    available_locations = ['B01.1', 'B02.1', 'B03.1', 'B04.1', 'B05.1', 'B06.1', 'B07.1']
    
    # Test seasonal optimized selection
    selected = allocation_service.select_optimal_location(available_locations, 'seasonal_optimized')
    print(f"Available B Rack 1 locations: {available_locations}")
    print(f"Selected location (seasonal_optimized): {selected}")
    
    if selected == 'B07.1':
        print("✅ Selected B07.1 - Highest seasonal priority location chosen!")
    else:
        print(f"❌ Selected {selected} instead of B07.1 - Seasonal optimization may not be working")
    
    # Test with P slots
    p_available = ['P01.1', 'P02.1', 'P07.1', 'P05.1']
    p_selected = allocation_service.select_optimal_location(p_available, 'seasonal_optimized')
    print(f"\nAvailable P locations: {p_available}")
    print(f"Selected P location (seasonal_optimized): {p_selected}")
    
    if p_selected == 'P07.1':
        print("✅ Selected P07.1 - Highest seasonal priority P location chosen!")
    else:
        print(f"❌ Selected {p_selected} instead of P07.1 - P slot optimization may not be working")
    
    # Test with D slots - should prefer closest to packing point
    d_available = ['D01.1', 'D05.1', 'D14.1', 'D08.1']
    d_selected = allocation_service.select_optimal_location(d_available, 'seasonal_optimized')
    print(f"\nAvailable D locations: {d_available}")
    print(f"Selected D location (seasonal_optimized): {d_selected}")
    
    if d_selected == 'D01.1':
        print("✅ Selected D01.1 - Closest D location to packing point chosen!")
    elif d_selected == 'D08.1':
        print("✅ Selected D08.1 - Also close to packing point, good choice!")
    else:
        print(f"❌ Selected {d_selected} - May not be optimizing for distance to packing point")

def test_rack_group_priority():
    """Test that rack groups now prioritize high-numbered slots"""
    print("\n🧪 Testing Rack Group Priority Order")
    print("=" * 60)
    
    # Test B Rack 1 filtering
    all_b_locations = ['B01.1', 'B02.1', 'B03.1', 'B04.1', 'B05.1', 'B06.1', 'B07.1']
    filtered = allocation_service.filter_locations_by_rack_group(all_b_locations, 'B Rack 1')
    
    print(f"All B locations: {all_b_locations}")
    print(f"Filtered for 'B Rack 1': {filtered}")
    print("Note: Filter should still include all B01-B07, but selection algorithm will prioritize B07")

if __name__ == "__main__":
    print("🚀 TESTING SEASONAL OPTIMIZATION IN ALLOCATION SERVICE")
    print("=" * 80)
    
    try:
        test_coordinate_mapping()
        test_location_selection()
        test_rack_group_priority()
        
        print("\n🎯 SUMMARY")
        print("=" * 80)
        print("✅ If all tests pass, the allocation service now:")
        print("   - Uses seasonal-optimized coordinates (B07/P07/D14 closest to packing)")
        print("   - Prioritizes high-numbered slots for storage allocation")
        print("   - Implements the business requirement: store in B07.1-4 first, then B06.1-4, etc.")
        print("\n🔄 Next step: Test with real storage operations to verify AI model allocations")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
