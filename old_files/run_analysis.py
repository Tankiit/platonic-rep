#!/usr/bin/env python3
"""
Simple launcher for comprehensive multi-model analysis
"""

import argparse
from run_comprehensive_analysis import run_comprehensive_analysis
from analysis_config import print_config, QUICK_MODE

def main():
    parser = argparse.ArgumentParser(description="Launch comprehensive multi-model analysis")
    
    parser.add_argument('--quick', action='store_true', 
                       help='Enable quick mode for testing (faster, limited analysis)')
    
    parser.add_argument('--config', action='store_true',
                       help='Show current configuration and exit')
    
    parser.add_argument('--models', nargs='+', 
                       help='Override models to analyze')
    
    parser.add_argument('--datasets', nargs='+',
                       help='Override datasets to analyze')
    
    parser.add_argument('--device', type=str, default=None,
                       help='Force specific device (cpu, mps, cuda)')
    
    args = parser.parse_args()
    
    # Show configuration if requested
    if args.config:
        print_config()
        return
    
    # Enable quick mode if requested
    if args.quick:
        QUICK_MODE['enabled'] = True
        print("🚀 Quick mode enabled for faster testing!")
        print("   - Limited models and datasets")
        print("   - Reduced sample count")
        print("   - Faster execution")
    
    # Show final configuration
    print_config()
    
    # Confirm before starting
    if not args.quick:
        response = input("\nProceed with full analysis? This may take 1-2 hours. (y/N): ")
        if response.lower() != 'y':
            print("Analysis cancelled.")
            return
    else:
        response = input("\nProceed with quick analysis? This will take ~10-15 minutes. (y/N): ")
        if response.lower() != 'y':
            print("Analysis cancelled.")
            return
    
    print("\n🎯 Starting comprehensive analysis...")
    print("Press Ctrl+C to interrupt at any time")
    
    try:
        # Run the analysis
        results, log = run_comprehensive_analysis()
        
        if results and log:
            print("\n🎉 Analysis completed successfully!")
            print(f"Results saved to: {log.get('output_dir', 'unknown')}")
        else:
            print("\n❌ Analysis failed or was interrupted")
            
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
