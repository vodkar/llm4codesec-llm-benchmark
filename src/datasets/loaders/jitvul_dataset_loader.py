#!/usr/bin/env python3
"""
JitVul Dataset Loader

This module provides dataset loading functionality for the JitVul benchmark,
which contains real-world vulnerability data with function pairs (vulnerable vs. non-vulnerable)
and associated call graph information.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmark.benchmark_framework import BenchmarkSample, DatasetLoader


class JitVulDatasetLoader:
    """Dataset loader for JitVul benchmark format."""
    
    def __init__(self, source_dir: str = "benchmarks/JitVul/data"):
        """
        Initialize the JitVul dataset loader.
        
        Args:
            source_dir: Path to the JitVul dataset directory
        """
        self.source_dir = Path(source_dir)
        self.logger = logging.getLogger(__name__)
    
    def load_dataset(
        self, 
        data_file: str,
        task_type: str = "binary",
        target_cwe: Optional[str] = None,
        use_call_graph: bool = True,
        max_samples: Optional[int] = None
    ) -> List[BenchmarkSample]:
        """
        Load JitVul dataset from JSONL file.
        
        Args:
            data_file: Path to the JitVul JSONL dataset file
            task_type: Type of task ("binary", "multiclass", "cwe_specific")
            target_cwe: Target CWE for cwe_specific task (e.g., "CWE-125")
            use_call_graph: Whether to include call graph context
            max_samples: Maximum number of samples to load
            
        Returns:
            List of BenchmarkSample objects
        """
        data_path = Path(data_file)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_file}")
        
        samples: List[BenchmarkSample] = []
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if max_samples and len(samples) >= max_samples:
                        break
                        
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        item = json.loads(line)
                        # Convert each item to samples based on task type
                        jitvul_samples = self._convert_jitvul_item_to_samples(
                            item, line_num, task_type, target_cwe, use_call_graph
                        )
                        samples.extend(jitvul_samples)
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                        continue
                    except Exception as e:
                        self.logger.warning(f"Error processing line {line_num}: {e}")
                        continue
        
        except Exception as e:
            raise RuntimeError(f"Error loading JitVul dataset: {e}")
        
        # Apply final sample limit if needed
        if max_samples and len(samples) > max_samples:
            samples = samples[:max_samples]
        
        self.logger.info(f"Loaded {len(samples)} samples from JitVul dataset")
        return samples
    
    def _convert_jitvul_item_to_samples(
        self, 
        item: Dict[str, Any], 
        line_num: int,
        task_type: str,
        target_cwe: Optional[str],
        use_call_graph: bool
    ) -> List[BenchmarkSample]:
        """
        Convert a single JitVul item into BenchmarkSample objects.
        
        Args:
            item: Raw JitVul data item
            line_num: Line number for ID generation
            task_type: Type of task being performed
            target_cwe: Target CWE for cwe_specific task
            use_call_graph: Whether to include call graph context
            
        Returns:
            List containing appropriate samples based on task type
        """
        samples: List[BenchmarkSample] = []
        
        # Extract basic information
        vuln_func = item.get("vulnerable_function_body", "")
        non_vuln_func = item.get("non_vulnerable_function_body", "")
        cwe = item.get("cwe", "")
        project = item.get("project", "unknown")
        func_hash = item.get("func_hash", "")
        
        if not vuln_func.strip() or not non_vuln_func.strip():
            return samples
        
        # Extract common metadata
        base_metadata: Dict[str, Any] = {
            "project": project,
            "cwe": cwe,
            "function_hash": func_hash,
            "source": "jitvul",
            "line_number": line_num
        }
        
        # Create samples based on task type
        if task_type == "binary":
            # Create both vulnerable and non-vulnerable samples
            vuln_sample = BenchmarkSample(
                id=f"jitvul_{line_num}_vulnerable",
                code=self._augment_code_with_context(vuln_func, item, use_call_graph),
                label="VULNERABLE",
                metadata={**base_metadata, "function_type": "vulnerable", "original_cwe": cwe},
                cwe_types=cwe if cwe else None,
                severity=self._get_cwe_severity(cwe)
            )
            
            non_vuln_sample = BenchmarkSample(
                id=f"jitvul_{line_num}_non_vulnerable", 
                code=self._augment_code_with_context(non_vuln_func, item, use_call_graph),
                label="NOT_VULNERABLE",
                metadata={**base_metadata, "function_type": "non_vulnerable"},
                cwe_types=None,
                severity=None
            )
            
            samples.extend([vuln_sample, non_vuln_sample])
            
        elif task_type == "multiclass":
            # Only include vulnerable samples with CWE labels
            if cwe:
                vuln_sample = BenchmarkSample(
                    id=f"jitvul_{line_num}_vulnerable",
                    code=self._augment_code_with_context(vuln_func, item, use_call_graph),
                    label=cwe,
                    metadata={**base_metadata, "function_type": "vulnerable"},
                    cwe_types=cwe,
                    severity=self._get_cwe_severity(cwe)
                )
                samples.append(vuln_sample)
                
        elif task_type == "cwe_specific":
            # Filter for specific CWE type
            if target_cwe and cwe == target_cwe:
                # Include both vulnerable (positive) and non-vulnerable (negative) for this CWE
                vuln_sample = BenchmarkSample(
                    id=f"jitvul_{line_num}_vulnerable",
                    code=self._augment_code_with_context(vuln_func, item, use_call_graph),
                    label="VULNERABLE",
                    metadata={**base_metadata, "function_type": "vulnerable", "target_cwe": target_cwe},
                    cwe_types=cwe,
                    severity=self._get_cwe_severity(cwe)
                )
                
                non_vuln_sample = BenchmarkSample(
                    id=f"jitvul_{line_num}_non_vulnerable",
                    code=self._augment_code_with_context(non_vuln_func, item, use_call_graph),
                    label="NOT_VULNERABLE", 
                    metadata={**base_metadata, "function_type": "non_vulnerable", "target_cwe": target_cwe},
                    cwe_types=None,
                    severity=None
                )
                
                samples.extend([vuln_sample, non_vuln_sample])
            
        return samples
    
    def _augment_code_with_context(self, code: str, item: Dict[str, Any], use_call_graph: bool) -> str:
        """
        Augment code with call graph context if available.
        
        Args:
            code: Original function code
            item: JitVul data item
            use_call_graph: Whether to add call graph context
            
        Returns:
            Augmented code string
        """
        if not use_call_graph or "call_graph" not in item:
            return code
        
        call_graph = item.get("call_graph", [])
        if not call_graph:
            return code
        
        # Add call graph context as comments
        context_lines = ["// Call graph context:"]
        for func_name in call_graph[:5]:  # Limit to first 5 for token efficiency
            context_lines.append(f"// - {func_name}")
        
        context = "\n".join(context_lines) + "\n\n"
        return context + code
    
    def _get_cwe_severity(self, cwe: list[str]) -> str:
        """
        Determine severity level for a CWE.
        
        Args:
            cwe: CWE identifier (e.g., "CWE-125")
            
        Returns:
            Severity level (HIGH, MEDIUM, LOW)
        """
        if not cwe:
            return "LOW"
        
        # Extract numeric part
        single_cwe = cwe[0] 
        cwe_num = single_cwe.replace("CWE-", "")
        
        # High severity CWEs
        high_severity_cwes = {
            "78",   # OS Command Injection
            "79",   # Cross-site Scripting
            "89",   # SQL Injection
            "94",   # Code Injection
            "352",  # CSRF
            "434",  # Unrestricted Upload
            "611"   # XML External Entities
        }
        
        # Medium severity CWEs
        medium_severity_cwes = {
            "125",  # Out-of-bounds Read
            "190",  # Integer Overflow
            "787",  # Out-of-bounds Write
            "476",  # NULL Pointer Dereference
            "416",  # Use After Free
            "502"   # Deserialization
        }
        
        if cwe_num in high_severity_cwes:
            return "HIGH"
        elif cwe_num in medium_severity_cwes:
            return "MEDIUM"
        else:
            return "LOW"
    
    def get_dataset_stats(self, data_file: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the dataset.
        
        Args:
            data_file: Path to the dataset file
            
        Returns:
            Dictionary containing dataset statistics
        """
        stats: Dict[str, Any] = {
            "total_items": 0,
            "cwe_distribution": defaultdict(int),
            "project_distribution": defaultdict(int),
            "severity_distribution": defaultdict(int),
            "has_call_graph": 0,
            "average_function_length": {"vulnerable": 0, "non_vulnerable": 0}
        }
        
        vuln_lengths: List[int] = []
        non_vuln_lengths: List[int] = []
        
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        item = json.loads(line)
                        stats["total_items"] += 1
                        
                        # CWE distribution
                        cwe = item.get("CWE", "Unknown")
                        stats["cwe_distribution"][cwe] += 1
                        
                        # Project distribution
                        project = item.get("project", "Unknown")
                        stats["project_distribution"][project] += 1
                        
                        # Severity distribution
                        severity = self._get_cwe_severity(cwe)
                        stats["severity_distribution"][severity] += 1
                        
                        # Call graph presence
                        if "call_graph" in item and item["call_graph"]:
                            stats["has_call_graph"] += 1
                        
                        # Function lengths
                        vuln_func = item.get("vulnerable_function", "")
                        non_vuln_func = item.get("non_vulnerable_function", "")
                        vuln_lengths.append(len(vuln_func))
                        non_vuln_lengths.append(len(non_vuln_func))
                        
                    except json.JSONDecodeError:
                        continue
        
        except Exception as e:
            self.logger.error(f"Error generating statistics: {e}")
            return {}
        
        # Calculate averages
        if vuln_lengths:
            stats["average_function_length"]["vulnerable"] = sum(vuln_lengths) / len(vuln_lengths)
        if non_vuln_lengths:
            stats["average_function_length"]["non_vulnerable"] = sum(non_vuln_lengths) / len(non_vuln_lengths)
        
        # Convert defaultdicts to regular dicts
        stats["cwe_distribution"] = dict(stats["cwe_distribution"])
        stats["project_distribution"] = dict(stats["project_distribution"])
        stats["severity_distribution"] = dict(stats["severity_distribution"])
        
        return stats


class JitVulDatasetLoaderFramework(DatasetLoader):
    """Framework-compatible dataset loader for JitVul."""
    
    def __init__(self, source_dir: str = "benchmarks/JitVul/data"):
        self.jitvul_loader = JitVulDatasetLoader(source_dir)
        self.logger = logging.getLogger(__name__)
    
    def load_dataset(self, path: str) -> List[BenchmarkSample]:
        """
        Load dataset using the framework interface.
        
        Args:
            path: Path to the dataset file
            
        Returns:
            List of BenchmarkSample objects
        """
        return self.jitvul_loader.load_dataset(
            data_file=path,
            task_type="binary",
            use_call_graph=True
        )
    
    def get_dataset_info(self, data_file: str) -> Dict[str, Any]:
        """
        Get information about the dataset.
        
        Args:
            data_file: Path to the dataset file
            
        Returns:
            Dictionary with dataset information
        """
        return self.jitvul_loader.get_dataset_stats(data_file)


def main():
    """Example usage and testing of the JitVul dataset loader."""
    loader = JitVulDatasetLoader()
    
    # Example dataset path (adjust as needed)
    dataset_path = Path(__file__).parent.parent.parent / "benchmarks" / "JitVul" / "data" / "final_benchmark.jsonl"
    
    if dataset_path.exists():
        try:
            # Test different task types
            print("Testing JitVul Dataset Loader")
            print("=" * 40)
            
            # Binary task
            print("\n1. Binary Classification:")
            binary_samples = loader.load_dataset(str(dataset_path), task_type="binary", max_samples=10)
            print(f"   Loaded {len(binary_samples)} binary samples")
            
            # Multiclass task
            print("\n2. Multiclass Classification:")
            multiclass_samples = loader.load_dataset(str(dataset_path), task_type="multiclass", max_samples=10)
            print(f"   Loaded {len(multiclass_samples)} multiclass samples")
            
            # CWE-specific task
            print("\n3. CWE-Specific Classification:")
            cwe_samples = loader.load_dataset(str(dataset_path), task_type="cwe_specific", target_cwe="CWE-125", max_samples=10)
            print(f"   Loaded {len(cwe_samples)} CWE-125 samples")
            
            # Get statistics
            print("\n4. Dataset Statistics:")
            stats = loader.get_dataset_stats(str(dataset_path))
            print(f"   Total items: {stats.get('total_items', 0)}")
            print(f"   CWE distribution: {list(stats.get('cwe_distribution', {}).keys())[:5]}")
            print(f"   Severity distribution: {stats.get('severity_distribution', {})}")
            
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Dataset file not found: {dataset_path}")
        print("Please ensure the JitVul dataset is available.")


if __name__ == "__main__":
    main()
