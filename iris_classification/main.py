"""
Main execution script for Iris Classification project
Run with: python main.py --mode [train/test/all]
"""

import argparse
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src import train, test


def main():
    parser = argparse.ArgumentParser(
        description='Iris Classification - Train and Test ML Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode train     # Train the model
  python main.py --mode test      # Test the model
  python main.py --mode all       # Train and test
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['train', 'test', 'all'],
        default='all',
        help='Mode of execution (default: all)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode in ['train', 'all']:
            print("\n" + "█"*70)
            print("█ STARTING TRAINING".ljust(70) + "█")
            print("█"*70)
            train.main()
            
            if args.mode == 'train':
                print("\nTraining complete! Run 'python main.py --mode test' to test the model.")
                return
        
        if args.mode in ['test', 'all']:
            print("\n" + "█"*70)
            print("█ STARTING TESTING".ljust(70) + "█")
            print("█"*70)
            test.main()
    
    except KeyboardInterrupt:
        print("\n\n⚠ Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
