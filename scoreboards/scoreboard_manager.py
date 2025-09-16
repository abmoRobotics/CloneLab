#!/usr/bin/env python3
"""
CloneLab Scoreboard Manager

This script provides automated management of the CloneLab scoreboard system,
allowing easy addition of new results and maintenance of the leaderboards.
"""

import argparse
import csv
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ScoreboardManager:
    """Manages the CloneLab scoreboard system."""
    
    def __init__(self, scoreboards_dir: str = None):
        """Initialize the scoreboard manager.
        
        Args:
            scoreboards_dir: Path to the scoreboards directory
        """
        if scoreboards_dir is None:
            # Get the directory where this script is located
            script_dir = Path(__file__).parent
            self.scoreboards_dir = script_dir
        else:
            self.scoreboards_dir = Path(scoreboards_dir)
            
        self.baselines_file = self.scoreboards_dir / "rlroverlab_baselines.md"
        self.results_file = self.scoreboards_dir / "clonelab_results.md"
    
    def _parse_markdown_table(self, file_path: Path) -> Tuple[List[str], List[List[str]]]:
        """Parse a markdown table and return headers and rows.
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            Tuple of (headers, rows)
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Scoreboard file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Find the table in markdown content
        lines = content.split('\n')
        table_start = -1
        table_end = -1
        
        for i, line in enumerate(lines):
            if '|' in line and 'Model Name' in line:
                table_start = i
                break
        
        if table_start == -1:
            raise ValueError(f"Could not find table in {file_path}")
        
        # Find table end (first line without |)
        for i in range(table_start + 2, len(lines)):  # Skip header separator
            if '|' not in lines[i].strip():
                table_end = i
                break
        
        if table_end == -1:
            table_end = len(lines)
        
        # Parse headers
        header_line = lines[table_start].strip()
        headers = [h.strip() for h in header_line.split('|')[1:-1]]  # Remove empty first/last
        
        # Parse rows
        rows = []
        for i in range(table_start + 2, table_end):  # Skip header and separator
            line = lines[i].strip()
            if line and '|' in line:
                row = [cell.strip() for cell in line.split('|')[1:-1]]  # Remove empty first/last
                if len(row) == len(headers):  # Only add complete rows
                    rows.append(row)
        
        return headers, rows
    
    def _write_markdown_table(self, file_path: Path, headers: List[str], rows: List[List[str]]):
        """Write headers and rows back to a markdown table file.
        
        Args:
            file_path: Path to the markdown file
            headers: Table headers
            rows: Table rows
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Scoreboard file not found: {file_path}")
        
        # Read the original file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Find the table section
        lines = content.split('\n')
        table_start = -1
        table_end = -1
        
        for i, line in enumerate(lines):
            if '|' in line and 'Model Name' in line:
                table_start = i
                break
        
        if table_start == -1:
            raise ValueError(f"Could not find table in {file_path}")
        
        # Find table end
        for i in range(table_start + 2, len(lines)):
            if '|' not in lines[i].strip():
                table_end = i
                break
        
        if table_end == -1:
            table_end = len(lines)
        
        # Build new table
        new_table_lines = []
        
        # Header row
        header_row = '| ' + ' | '.join(headers) + ' |'
        new_table_lines.append(header_row)
        
        # Separator row
        separator = '|' + ''.join(['-' * (len(h) + 2) + '|' for h in headers])
        new_table_lines.append(separator)
        
        # Data rows
        for row in rows:
            data_row = '| ' + ' | '.join(row) + ' |'
            new_table_lines.append(data_row)
        
        # Reconstruct the file
        new_content = (
            '\n'.join(lines[:table_start]) +
            '\n' + '\n'.join(new_table_lines) + '\n' +
            '\n'.join(lines[table_end:])
        )
        
        # Write back to file
        with open(file_path, 'w') as f:
            f.write(new_content)
    
    def add_baseline_result(
        self, 
        name: str, 
        algorithm: str, 
        environment: str, 
        reward: float, 
        success_rate: float, 
        notes: str = ""
    ):
        """Add a new baseline result to the RLRoverLab baselines scoreboard.
        
        Args:
            name: Model name
            algorithm: Algorithm used (e.g., SAC, BC, CQL)
            environment: Environment name
            reward: Average reward
            success_rate: Success rate (0-1)
            notes: Additional notes
        """
        try:
            headers, rows = self._parse_markdown_table(self.baselines_file)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error reading baselines file: {e}")
            return False
        
        # Create new row
        new_row = [
            name,
            "Baseline",
            algorithm,
            environment,
            "Reward",
            str(int(reward)),
            f"{success_rate:.2f}",
            notes
        ]
        
        # Check if model already exists
        for i, row in enumerate(rows):
            if row[0] == name:
                print(f"Model '{name}' already exists. Updating...")
                rows[i] = new_row
                break
        else:
            # Add new row
            rows.append(new_row)
        
        # Sort by score (descending)
        rows.sort(key=lambda x: float(x[5]), reverse=True)
        
        try:
            self._write_markdown_table(self.baselines_file, headers, rows)
            print(f"Successfully added baseline result for '{name}'")
            return True
        except Exception as e:
            print(f"Error writing to baselines file: {e}")
            return False
    
    def add_student_result(
        self,
        name: str,
        algorithm: str,
        environment: str,
        reward: float,
        success_rate: float,
        notes: str = ""
    ):
        """Add a new student result to the CloneLab results scoreboard.
        
        Args:
            name: Model name
            algorithm: Algorithm used (e.g., IQL, BC, CQL)
            environment: Environment name
            reward: Average reward
            success_rate: Success rate (0-1)
            notes: Additional notes
        """
        try:
            headers, rows = self._parse_markdown_table(self.results_file)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error reading results file: {e}")
            return False
        
        # Create new row
        new_row = [
            name,
            "Student",
            algorithm,
            environment,
            "Reward",
            str(int(reward)),
            f"{success_rate:.2f}",
            notes
        ]
        
        # Check if model already exists
        for i, row in enumerate(rows):
            if row[0] == name:
                print(f"Model '{name}' already exists. Updating...")
                rows[i] = new_row
                break
        else:
            # Add new row
            rows.append(new_row)
        
        # Sort by score (descending)
        rows.sort(key=lambda x: float(x[5]), reverse=True)
        
        try:
            self._write_markdown_table(self.results_file, headers, rows)
            print(f"Successfully added student result for '{name}'")
            return True
        except Exception as e:
            print(f"Error writing to results file: {e}")
            return False
    
    def list_results(self, scoreboard: str = "both"):
        """List current results from scoreboards.
        
        Args:
            scoreboard: Which scoreboard to show ("baselines", "results", or "both")
        """
        if scoreboard in ["baselines", "both"]:
            print("\n=== RLRoverLab Baselines ===")
            try:
                headers, rows = self._parse_markdown_table(self.baselines_file)
                print(f"{'Model Name':<20} {'Algorithm':<10} {'Environment':<15} {'Score':<8} {'SR':<6}")
                print("-" * 65)
                for row in rows:
                    print(f"{row[0]:<20} {row[2]:<10} {row[3]:<15} {row[5]:<8} {row[6]:<6}")
            except Exception as e:
                print(f"Error reading baselines: {e}")
        
        if scoreboard in ["results", "both"]:
            print("\n=== CloneLab Results ===")
            try:
                headers, rows = self._parse_markdown_table(self.results_file)
                print(f"{'Model Name':<20} {'Algorithm':<10} {'Environment':<15} {'Score':<8} {'SR':<6}")
                print("-" * 65)
                for row in rows:
                    print(f"{row[0]:<20} {row[2]:<10} {row[3]:<15} {row[5]:<8} {row[6]:<6}")
            except Exception as e:
                print(f"Error reading results: {e}")
    
    def export_csv(self, output_file: str, scoreboard: str = "both"):
        """Export scoreboard data to CSV.
        
        Args:
            output_file: Output CSV file path
            scoreboard: Which scoreboard to export ("baselines", "results", or "both")
        """
        all_rows = []
        
        if scoreboard in ["baselines", "both"]:
            try:
                headers, rows = self._parse_markdown_table(self.baselines_file)
                for row in rows:
                    row_data = dict(zip(headers, row))
                    row_data['Scoreboard'] = 'RLRoverLab'
                    all_rows.append(row_data)
            except Exception as e:
                print(f"Error reading baselines: {e}")
        
        if scoreboard in ["results", "both"]:
            try:
                headers, rows = self._parse_markdown_table(self.results_file)
                for row in rows:
                    row_data = dict(zip(headers, row))
                    row_data['Scoreboard'] = 'CloneLab'
                    all_rows.append(row_data)
            except Exception as e:
                print(f"Error reading results: {e}")
        
        if all_rows:
            # Get all possible field names
            all_fields = set()
            for row in all_rows:
                all_fields.update(row.keys())
            
            fieldnames = ['Scoreboard'] + [f for f in headers if f in all_fields]
            
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)
            
            print(f"Exported {len(all_rows)} results to {output_file}")
        else:
            print("No data to export")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="CloneLab Scoreboard Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add baseline result
    baseline_parser = subparsers.add_parser('add-baseline', help='Add a baseline result')
    baseline_parser.add_argument('--name', required=True, help='Model name')
    baseline_parser.add_argument('--algorithm', required=True, help='Algorithm (e.g., SAC, BC)')
    baseline_parser.add_argument('--environment', required=True, help='Environment name')
    baseline_parser.add_argument('--reward', type=float, required=True, help='Average reward')
    baseline_parser.add_argument('--success-rate', type=float, required=True, help='Success rate (0-1)')
    baseline_parser.add_argument('--notes', default='', help='Additional notes')
    
    # Add student result
    result_parser = subparsers.add_parser('add-result', help='Add a student result')
    result_parser.add_argument('--name', required=True, help='Model name')
    result_parser.add_argument('--algorithm', required=True, help='Algorithm (e.g., IQL, BC)')
    result_parser.add_argument('--environment', required=True, help='Environment name')
    result_parser.add_argument('--reward', type=float, required=True, help='Average reward')
    result_parser.add_argument('--success-rate', type=float, required=True, help='Success rate (0-1)')
    result_parser.add_argument('--notes', default='', help='Additional notes')
    
    # List results
    list_parser = subparsers.add_parser('list', help='List current results')
    list_parser.add_argument('--scoreboard', choices=['baselines', 'results', 'both'], 
                           default='both', help='Which scoreboard to show')
    
    # Export to CSV
    export_parser = subparsers.add_parser('export', help='Export results to CSV')
    export_parser.add_argument('--output', required=True, help='Output CSV file')
    export_parser.add_argument('--scoreboard', choices=['baselines', 'results', 'both'], 
                             default='both', help='Which scoreboard to export')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize manager
    manager = ScoreboardManager()
    
    # Execute commands
    if args.command == 'add-baseline':
        manager.add_baseline_result(
            args.name, args.algorithm, args.environment,
            args.reward, args.success_rate, args.notes
        )
    
    elif args.command == 'add-result':
        manager.add_student_result(
            args.name, args.algorithm, args.environment,
            args.reward, args.success_rate, args.notes
        )
    
    elif args.command == 'list':
        manager.list_results(args.scoreboard)
    
    elif args.command == 'export':
        manager.export_csv(args.output, args.scoreboard)


if __name__ == '__main__':
    main()