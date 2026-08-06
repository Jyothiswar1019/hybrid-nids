import os
import sys
import time
import socket
from collections import defaultdict, Counter
from scapy.all import sniff, IP, TCP, UDP, Raw

sys.path.append('src')
from predict import predict_sample

# ===== SERVICE MAPPING =====
def get_service(port):
    """Map port to NSL-KDD service names."""
    # NSL-KDD known services (common ones)
    service_map = {
        80: 'http',
        443: 'http',         # map https to http
        8080: 'http',
        21: 'ftp',
        23: 'telnet',
        25: 'smtp',
        110: 'pop_3',
        139: 'netbios_ssn',
        445: 'microsoft_ds',
        22: 'ssh',
        53: 'domain',
        143: 'imap',
        993: 'imaps',
        995: 'pop3s',
        5432: 'postgres',
        3306: 'mysql',
        3389: 'ms_terminal_services',
        5900: 'vnc',
        137: 'netbios_ns',
        138: 'netbios_dgm',
        389: 'ldap',
        636: 'ldaps',
        25: 'smtp',
        587: 'smtp',
        465: 'smtps',
        110: 'pop_3',
        995: 'pop3s',
        993: 'imaps',
        143: 'imap',
        123: 'ntp',
        514: 'syslog',
        # add more if needed
    }
    return service_map.get(port, 'other')

def get_flag(tcp_pkt):
    """Map TCP flags to NSL-KDD flag codes."""
    if tcp_pkt is None:
        return 'SF'
    flags = tcp_pkt.flags
    if flags == 0x02:          # SYN
        return 'S0'
    elif flags == 0x12:        # SYN-ACK
        return 'S1'
    elif flags == 0x04:        # RST
        return 'RSTO'
    elif flags == 0x10:        # ACK
        return 'SF'            # treat as established
    elif flags == 0x14:        # RST-ACK
        return 'RSTO'
    else:
        return 'SF'

def extract_features_from_packets(packets, window_seconds=10):
    if not packets:
        return None

    connections = defaultdict(lambda: {'count': 0, 'src_bytes': 0, 'dst_bytes': 0, 'service': 'other', 'flag': 'SF'})
    dst_host_counts = Counter()
    service_counts = Counter()
    total_connections = 0

    for pkt in packets:
        if IP not in pkt:
            continue
        ip = pkt[IP]
        if TCP not in pkt:
            continue
        tcp = pkt[TCP]

        dst_ip = ip.dst
        dport = tcp.dport
        sport = tcp.sport

        # Use destination port for service (or source port if dest is ephemeral? Usually dest port is the service port)
        service = get_service(dport)
        # If service is 'other' and dport > 1024, try source port? This is simplified.
        if service == 'other' and dport > 1024:
            service = get_service(sport)   # maybe the source port is the service port

        key = (dst_ip, dport)
        connections[key]['count'] += 1
        total_connections += 1
        connections[key]['src_bytes'] += len(pkt)  # rough size
        connections[key]['service'] = service
        service_counts[service] += 1

        flag = get_flag(tcp)
        connections[key]['flag'] = flag

        dst_host_counts[dst_ip] += 1

    if total_connections == 0:
        return None

    # Pick the most frequent connection as representative
    # Actually we need to aggregate per connection; we'll take the first key for simplicity
    first_key = list(connections.keys())[0]
    dst_ip, dport = first_key
    conn = connections[first_key]

    count = conn['count']
    same_srv_rate = service_counts[conn['service']] / total_connections if total_connections > 0 else 0.0
    diff_srv_rate = 1.0 - same_srv_rate

    dst_host_srv_count = sum(1 for k, v in connections.items() if k[0] == dst_ip)
    dst_host_same_srv_rate = sum(1 for k, v in connections.items() if k[0] == dst_ip and k[1] == dport) / dst_host_srv_count if dst_host_srv_count > 0 else 0.0
    dst_host_diff_srv_rate = 1.0 - dst_host_same_srv_rate

    serror_count = sum(1 for k, v in connections.items() if v['flag'] in ['S0', 'RSTO'])
    dst_host_serror_rate = serror_count / dst_host_srv_count if dst_host_srv_count > 0 else 0.0

    features = {
        'service': conn['service'],
        'flag': conn['flag'],
        'src_bytes': conn['src_bytes'],
        'count': count,
        'same_srv_rate': same_srv_rate,
        'diff_srv_rate': diff_srv_rate,
        'dst_host_srv_count': dst_host_srv_count,
        'dst_host_same_srv_rate': dst_host_same_srv_rate,
        'dst_host_diff_srv_rate': dst_host_diff_srv_rate,
        'dst_host_serror_rate': dst_host_serror_rate
    }
    return features

def sniff_live(interface=None, duration=10):
    print(f"👂 Sniffing traffic on interface '{interface or 'default'}' for {duration} seconds...")
    print("   (Generate traffic from your phone to laptop, e.g., browse a website, ping, etc.)")
    packets = sniff(iface=interface, timeout=duration, count=0)
    print(f"✅ Sniffed {len(packets)} packets.")
    return packets

def main():
    interface = "Wi-Fi"  # or None for auto

    # Ask user for duration
    try:
        duration = int(input("Enter sniff duration in seconds (e.g., 10, 30, 60): "))
    except ValueError:
        duration = 10
        print("Invalid input. Using default 10 seconds.")

    packets = sniff_live(interface=interface, duration=duration)
    if not packets:
        print("❌ No packets captured.")
        return

    features = extract_features_from_packets(packets, window_seconds=duration)
    if not features:
        print("❌ Could not extract features.")
        return

    print("\n📊 Extracted Features:")
    for k, v in features.items():
        print(f"   {k}: {v}")

    pred, conf, ptype = predict_sample(features)
    print("\n" + "="*50)
    print(f"🎯 Prediction: {pred}")
    print(f"📈 Confidence: {conf:.4f}")
    print(f"🏷️  Type: {ptype}")
    print("="*50)
if __name__ == "__main__":
    main()
