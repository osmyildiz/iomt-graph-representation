"""
PCAP IP/Port/Timestamp Extraction
====================================
Extracts source IP, destination IP, source port, destination port,
timestamp, and protocol from raw PCAP files using tshark.

This script was run on Mac (not DGX) due to PCAP file locality.
Produces CSV files with columns: timestamp, src_ip, dst_ip, src_port, dst_port, protocol

Input: Directory of PCAP files from CICIoMT2024
Output: Directory of CSV files with extracted fields

Usage:
    python src/pcap_extraction.py --pcap_dir /path/to/pcaps --output_dir /path/to/ip_extracted

Requirements:
    - tshark (Wireshark CLI) must be installed
    - PCAP files from CICIoMT2024 WiFi_and_MQTT/attacks/pcap/
"""

import os
import subprocess
import argparse
from pathlib import Path


def extract_pcap(pcap_path, output_path):
    """
    Extract IP/port/timestamp/protocol from a single PCAP file using tshark.
    
    Extracts fields:
        - frame.time_epoch: Unix timestamp
        - ip.src: Source IP address
        - ip.dst: Destination IP address  
        - tcp.srcport/udp.srcport: Source port
        - tcp.dstport/udp.dstport: Destination port
        - _ws.col.Protocol: Protocol name
    """
    cmd = [
        'tshark',
        '-r', pcap_path,
        '-T', 'fields',
        '-e', 'frame.time_epoch',
        '-e', 'ip.src',
        '-e', 'ip.dst',
        '-e', 'tcp.srcport',
        '-e', 'udp.srcport',
        '-e', 'tcp.dstport',
        '-e', 'udp.dstport',
        '-e', '_ws.col.Protocol',
        '-E', 'header=y',
        '-E', 'separator=,',
        '-E', 'quote=d',
        '-E', 'occurrence=f',
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  ERROR processing {pcap_path}: {result.stderr[:200]}")
            return False

        with open(output_path, 'w') as f:
            lines = result.stdout.strip().split('\n')
            # Write header
            f.write('timestamp,src_ip,dst_ip,src_port,dst_port,protocol\n')
            
            for line in lines[1:]:  # Skip tshark header
                parts = line.split(',')
                if len(parts) < 8:
                    continue
                
                timestamp = parts[0].strip('"')
                src_ip = parts[1].strip('"')
                dst_ip = parts[2].strip('"')
                
                # Merge TCP/UDP source port
                tcp_sport = parts[3].strip('"')
                udp_sport = parts[4].strip('"')
                src_port = tcp_sport if tcp_sport else udp_sport
                
                # Merge TCP/UDP destination port
                tcp_dport = parts[5].strip('"')
                udp_dport = parts[6].strip('"')
                dst_port = tcp_dport if tcp_dport else udp_dport
                
                protocol = parts[7].strip('"')
                
                # Skip rows without IP addresses
                if not src_ip or not dst_ip:
                    continue
                
                f.write(f'{timestamp},{src_ip},{dst_ip},{src_port},{dst_port},{protocol}\n')
        
        return True
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT processing {pcap_path}")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Extract IP/port/timestamp from PCAP files')
    parser.add_argument('--pcap_dir', required=True, help='Directory containing PCAP files')
    parser.add_argument('--output_dir', required=True, help='Output directory for extracted CSVs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pcap_files = sorted([
        f for f in os.listdir(args.pcap_dir)
        if f.endswith('.pcap') or f.endswith('.pcapng')
    ])

    print(f"Found {len(pcap_files)} PCAP files in {args.pcap_dir}")
    print(f"Output directory: {args.output_dir}")

    success, fail = 0, 0
    for i, pcap_file in enumerate(pcap_files):
        pcap_path = os.path.join(args.pcap_dir, pcap_file)
        output_file = Path(pcap_file).stem + '_extracted.csv'
        output_path = os.path.join(args.output_dir, output_file)

        # Skip if already extracted
        if os.path.exists(output_path):
            print(f"  [{i+1}/{len(pcap_files)}] SKIP (exists): {pcap_file}")
            success += 1
            continue

        file_size_mb = os.path.getsize(pcap_path) / (1024 * 1024)
        print(f"  [{i+1}/{len(pcap_files)}] Processing: {pcap_file} ({file_size_mb:.1f} MB)")

        if extract_pcap(pcap_path, output_path):
            # Count extracted rows
            with open(output_path) as f:
                n_rows = sum(1 for _ in f) - 1  # minus header
            print(f"    → {n_rows:,} packets extracted")
            success += 1
        else:
            fail += 1

    print(f"\nDone: {success} success, {fail} failed out of {len(pcap_files)}")


if __name__ == "__main__":
    main()
