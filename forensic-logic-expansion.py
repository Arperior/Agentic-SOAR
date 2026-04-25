def _generate_forensic_justification(self, row, threat):
        """
        Forensic Mapping based on UNSW-NB15 Attack Categories
        """
        reasons = []
        val = lambda x: row[x].values[0]

        # 1. DOS & FUZZERS (High Load / Low Duration)
        if threat in ['DoS', 'Fuzzers']:
            if val('sload') > 500000 or val('dload') > 500000:
                reasons.append(f"Abnormal Load detected (S:{val('sload'):.0f}/D:{val('dload'):.0f} bps).")
            if val('dur') < 0.1:
                reasons.append("High-frequency burst pattern identified.")

        # 2. RECONNAISSANCE (Mass Scanning)
        elif threat == 'Reconnaissance':
            if val('ct_srv_src') > 10 or val('ct_dst_ltm') > 10:
                reasons.append(f"Scanner behavior: {val('ct_srv_src')} connections to same service.")
            if val('is_sm_ips_ports') == 1:
                reasons.append("Identical Source/Destination IP and Port detected (Spoofing).")

        # 3. EXPLOITS & SHELLCODE (Buffer/TTL anomalies)
        elif threat in ['Exploits', 'Shellcode']:
            if val('sttl') in [252, 254, 255]:
                reasons.append(f"Suspicious TTL ({val('sttl')}): Likely crafted packet for OS bypass.")
            if val('smeansz') > 500:
                reasons.append("Large mean packet size suggests payload injection attempt.")

        # 4. BACKDOORS (Persistence / Packet Loss)
        elif threat == 'Backdoor':
            if val('sloss') > 0 or val('dloss') > 0:
                reasons.append("Elevated packet loss suggests an unstable or hidden tunnel.")
            if val('service') == 'unknown' and val('is_ftp_login') == 0:
                reasons.append("Non-standard service usage with high persistence.")

        # 5. WORMS (Rapid Spreading)
        elif threat == 'Worms':
            if val('ct_src_dport_ltm') > 5 and val('ct_dst_sport_ltm') > 5:
                reasons.append("Rapid cross-port propagation behavior detected.")

        # 6. GENERIC / ANALYSIS
        else:
            reasons.append("Behavioral anomalies detected in flow state transitions.")

        return f"Forensic Audit for {threat}: " + (" ".join(reasons) if reasons else "Pattern matches statistical anomaly baseline.")