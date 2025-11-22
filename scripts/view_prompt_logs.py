"""
scripts/view_prompt_logs.py

Interactive prompt log viewer
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional


class PromptLogViewer:
    """Interactive viewer for prompt logs"""
    
    def __init__(self, logs_dir: str = "logs/prompts"):
        self.base_dir = Path(logs_dir)
        
        if not self.base_dir.exists():
            print(f"❌ Logs directory not found: {self.base_dir}")
            sys.exit(1)
        
        self.astrosage_dir = self.base_dir / "astrosage"
        self.rewrite_dir = self.base_dir / "rewrite"
    
    def list_recent_logs(self, agent: str = "all", limit: int = 10):
        """List recent log files"""
        
        print("\n" + "="*80)
        print(f"RECENT PROMPT LOGS (Last {limit})")
        print("="*80 + "\n")
        
        logs = []
        
        if agent in ["all", "astrosage"]:
            astrosage_logs = list(self.astrosage_dir.glob("*.json"))
            logs.extend([("astrosage", f) for f in astrosage_logs])
        
        if agent in ["all", "rewrite"]:
            rewrite_logs = list(self.rewrite_dir.glob("*.json"))
            logs.extend([("rewrite", f) for f in rewrite_logs])
        
        # Sort by modification time (newest first)
        logs.sort(key=lambda x: x[1].stat().st_mtime, reverse=True)
        
        # Take only the last N
        logs = logs[:limit]
        
        if not logs:
            print("No logs found.")
            return []
        
        for i, (agent_name, log_file) in enumerate(logs, 1):
            mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
            print(f"{i:2d}. [{agent_name:10s}] {log_file.name}")
            print(f"    Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            print()
        
        return logs
    
    def view_log(self, log_file: Path):
        """View a specific log file"""
        
        # Check for text file first (human-readable)
        txt_file = log_file.with_suffix('.txt')
        
        if txt_file.exists():
            print("\n" + "="*80)
            print(f"VIEWING: {log_file.name}")
            print("="*80 + "\n")
            
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print(content)
        else:
            # Fallback to JSON
            print("\n" + "="*80)
            print(f"VIEWING (JSON): {log_file.name}")
            print("="*80 + "\n")
            
            with open(log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(json.dumps(data, indent=2, ensure_ascii=False))
    
    def view_log_summary(self, log_file: Path):
        """View summary of a log file"""
        
        with open(log_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("\n" + "="*80)
        print(f"LOG SUMMARY: {log_file.name}")
        print("="*80 + "\n")
        
        print(f"Agent: {data['agent']}")
        print(f"Timestamp: {data['timestamp']}")
        print(f"Session ID: {data['session_id']}")
        
        if data['agent'] == 'astrosage':
            print(f"Routing Strategy: {data['routing_strategy']}")
            print(f"Expertise Level: {data['expertise_level']}")
        elif data['agent'] == 'rewrite':
            print(f"Model Tier: {data['model_tier']}")
        
        # Statistics
        stats = data['statistics']
        print(f"\nPrompt Statistics:")
        print(f"  - System: {stats['system_prompt_length']:,} chars")
        print(f"  - User: {stats['user_message_length']:,} chars")
        print(f"  - Total: {stats['total_prompt_length']:,} chars")
        print(f"  - Estimated tokens: ~{stats['estimated_tokens']:,}")
        
        # Response (if available)
        if 'response' in data:
            resp = data['response']
            print(f"\nResponse:")
            print(f"  - Model: {resp['model_used']}")
            print(f"  - Tokens: {resp['tokens_used']}")
            print(f"  - Time: {resp['response_time']:.2f}s")
            
            if data['agent'] == 'astrosage':
                print(f"  - Success: {resp['success']}")
                if resp.get('error'):
                    print(f"  - Error: {resp['error']}")
            elif data['agent'] == 'rewrite':
                print(f"  - Validation: {'✅ PASSED' if resp['validation_passed'] else '❌ FAILED'}")
                if resp.get('validation_errors'):
                    print(f"  - Errors: {', '.join(resp['validation_errors'])}")
                if resp['retry_count'] > 0:
                    print(f"  - Retries: {resp['retry_count']}")
            
            print(f"  - Length: {resp['response_length']:,} chars ({resp['response_word_count']:,} words)")
        
        print()
    
    def search_logs(self, query: str, agent: str = "all"):
        """Search logs by content"""
        
        print(f"\n🔍 Searching for: '{query}'...\n")
        
        logs = []
        
        if agent in ["all", "astrosage"]:
            logs.extend(self.astrosage_dir.glob("*.json"))
        
        if agent in ["all", "rewrite"]:
            logs.extend(self.rewrite_dir.glob("*.json"))
        
        matches = []
        
        for log_file in logs:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read().lower()
                
                if query.lower() in content:
                    matches.append(log_file)
        
        if not matches:
            print("No matches found.")
            return []
        
        print(f"Found {len(matches)} matching log(s):\n")
        
        for i, log_file in enumerate(matches, 1):
            mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
            agent_name = "astrosage" if "astrosage" in str(log_file) else "rewrite"
            print(f"{i:2d}. [{agent_name:10s}] {log_file.name}")
            print(f"    Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            print()
        
        return matches
    
    def compare_params(self, log_file: Path):
        """Compare parameters in prompt vs response"""
        
        with open(log_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if data['agent'] != 'astrosage':
            print("❌ This feature is only for AstroSage logs")
            return
        
        print("\n" + "="*80)
        print("PARAMETER COMPARISON")
        print("="*80 + "\n")
        
        # Extract expected params
        analysis_results = data.get('analysis_results')
        
        if not analysis_results:
            print("No analysis results in this log.")
            return
        
        print("Expected Parameters (from prompt):")
        for model_name, params in analysis_results.items():
            print(f"\n  {model_name}:")
            for param, value in params.items():
                print(f"    {param} = {value:.6e}" if isinstance(value, float) else f"    {param} = {value}")
        
        # Check if response exists
        if 'response' not in data:
            print("\n⚠️  No response logged yet.")
            return
        
        # Extract params from response
        response_text = data['response']['content']
        
        print("\n" + "-"*80)
        print("Parameters Found in Response:")
        print("-"*80 + "\n")
        
        import re
        
        patterns = {
            'A': r'A\s*=\s*([0-9.]+(?:[eE][+-]?[0-9]+)?)',
            'b': r'\bb\s*=\s*([0-9.]+)',
            'fb': r'f_?b\s*=\s*([0-9.]+(?:[eE][+-]?[0-9]+)?)',
            'sh': r'(?:sh|α|alpha)\s*=\s*([0-9.]+)',
            'n': r'\bn\s*=\s*([0-9.]+(?:[eE][+-]?[0-9]+)?)'
        }
        
        found_params = {}
        for param_name, pattern in patterns.items():
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            if matches:
                found_params[param_name] = [float(m) for m in matches]
        
        if not found_params:
            print("❌ No parameters found in response!")
            return
        
        for param, values in found_params.items():
            print(f"  {param}: {values}")
        
        # Compare
        print("\n" + "-"*80)
        print("Comparison:")
        print("-"*80 + "\n")
        
        has_errors = False
        
        for model_name, expected_params in analysis_results.items():
            for param_name, expected_value in expected_params.items():
                if param_name not in found_params:
                    print(f"⚠️  {param_name}: NOT FOUND in response (expected {expected_value:.6e})")
                    has_errors = True
                else:
                    for found_value in found_params[param_name]:
                        rel_diff = abs(found_value - expected_value) / abs(expected_value)
                        
                        if rel_diff > 0.01:  # 1% tolerance
                            print(f"❌ {param_name}: MISMATCH - found {found_value:.6e}, expected {expected_value:.6e} (diff: {rel_diff*100:.1f}%)")
                            has_errors = True
                        else:
                            print(f"✅ {param_name}: OK - {found_value:.6e}")
        
        if not has_errors:
            print("\n✅ All parameters match!")
    
    def interactive(self):
        """Interactive mode"""
        
        while True:
            print("\n" + "="*80)
            print("PROMPT LOG VIEWER")
            print("="*80)
            print("\n1. List recent logs")
            print("2. View log (full)")
            print("3. View log (summary)")
            print("4. Search logs")
            print("5. Compare parameters (AstroSage)")
            print("6. Exit")
            
            choice = input("\nSelect option: ").strip()
            
            if choice == '1':
                agent = input("Agent (all/astrosage/rewrite) [all]: ").strip() or "all"
                limit = input("Limit [10]: ").strip()
                limit = int(limit) if limit else 10
                
                logs = self.list_recent_logs(agent=agent, limit=limit)
                
                if logs:
                    num = input("\nView log # (or Enter to skip): ").strip()
                    if num.isdigit():
                        idx = int(num) - 1
                        if 0 <= idx < len(logs):
                            self.view_log(logs[idx][1])
            
            elif choice == '2':
                agent = input("Agent (astrosage/rewrite): ").strip()
                filename = input("Filename: ").strip()
                
                if agent == 'astrosage':
                    log_file = self.astrosage_dir / filename
                else:
                    log_file = self.rewrite_dir / filename
                
                if log_file.exists():
                    self.view_log(log_file)
                else:
                    print(f"❌ File not found: {log_file}")
            
            elif choice == '3':
                agent = input("Agent (astrosage/rewrite): ").strip()
                filename = input("Filename: ").strip()
                
                if agent == 'astrosage':
                    log_file = self.astrosage_dir / filename
                else:
                    log_file = self.rewrite_dir / filename
                
                if log_file.exists():
                    self.view_log_summary(log_file)
                else:
                    print(f"❌ File not found: {log_file}")
            
            elif choice == '4':
                query = input("Search query: ").strip()
                agent = input("Agent (all/astrosage/rewrite) [all]: ").strip() or "all"
                
                matches = self.search_logs(query, agent=agent)
                
                if matches:
                    num = input("\nView log # (or Enter to skip): ").strip()
                    if num.isdigit():
                        idx = int(num) - 1
                        if 0 <= idx < len(matches):
                            self.view_log(matches[idx])
            
            elif choice == '5':
                filename = input("AstroSage log filename: ").strip()
                log_file = self.astrosage_dir / filename
                
                if log_file.exists():
                    self.compare_params(log_file)
                else:
                    print(f"❌ File not found: {log_file}")
            
            elif choice == '6':
                print("Goodbye!")
                break
            
            else:
                print("❌ Invalid option")


if __name__ == "__main__":
    viewer = PromptLogViewer()
    viewer.interactive()