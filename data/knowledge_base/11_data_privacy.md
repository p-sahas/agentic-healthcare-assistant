# Nawaloka Hospital — Data Privacy & Information Security Policy

## 1. Scope & Legal Framework

This policy applies to all employees, contractors, students, and volunteers who access, process, or store patient data or hospital information systems. It is aligned with:

- **Sri Lanka Personal Data Protection Act No. 9 of 2022 (PDPA).**
- **Sri Lanka Computer Crimes Act No. 24 of 2007.**
- **HIPAA principles** (adopted as international best practice for healthcare data).
- **ISO 27001:2022** — Information Security Management System (Nawaloka is ISO 27001 certified).

Violation of this policy may result in disciplinary action up to and including termination, and may also constitute a criminal offence.

## 2. Patient Data Classification

| Classification | Description | Examples | Access Control |
|---------------|-------------|----------|----------------|
| Highly Confidential | Sensitive health data requiring maximum protection | HIV/AIDS status, psychiatric records, genetic data, substance abuse records | Named-access only; break-the-glass for emergencies |
| Confidential | Standard patient health information | Diagnoses, lab results, imaging reports, medications, billing records | Role-based access (treating team + billing) |
| Internal | Hospital operational data not directly patient-identifiable | Aggregate statistics, departmental reports, staff rosters | All hospital staff |
| Public | Information intended for public disclosure | Hospital address, visiting hours, service catalogue, published research | Unrestricted |

## 3. Electronic Medical Record (EMR) Access

### 3.1 Access Levels

The HIS (Hospital Information System) enforces role-based access control (RBAC):

| Role | Access Scope |
|------|-------------|
| Consultant | Full access to their own patients; read-only for cross-departmental consults |
| Medical Officer | Read-write for assigned ward/unit patients |
| Nursing Staff | Read-write for nursing documentation; read-only for physician notes |
| Pharmacist | Read-write for medication orders and dispensing records |
| Lab Technologist | Read-write for lab results; read-only for clinical context |
| Billing Clerk | Read-only for billing-related fields (diagnoses codes, procedures, charges) |
| Admin / Reception | Patient demographics and appointment scheduling only |
| Medical Student / Intern | Read-only for assigned patients under supervision |

### 3.2 Authentication

- **Username + Password**: Minimum 12 characters, complexity requirements (upper, lower, digit, special character), changed every 90 days.
- **Two-Factor Authentication (2FA)**: Mandatory for all remote access (VPN, telemedicine platform) and for accessing Highly Confidential records.
- **Session Timeout**: Clinical workstations auto-lock after 5 minutes of inactivity. Users must re-authenticate.
- **Single Sign-On (SSO)**: Available within the hospital network for HIS, PACS, LIS, and email.

### 3.3 Break-the-Glass (BTG)

In an emergency where the treating physician does not have pre-authorised access to a patient's record, the BTG function allows temporary access. BTG events are:
- Logged with the user ID, patient HN, timestamp, and justification.
- Reviewed by the Privacy Officer within 24 hours.
- Flagged as a policy violation if the justification is insufficient.

## 4. Consent & Patient Rights

### 4.1 Consent for Data Processing

Patients sign a **General Consent for Treatment and Data Processing** (Form PRI-01) at first registration, covering:
- Collection and storage of health data for treatment purposes.
- Sharing with the treating team, pharmacy, laboratory, and billing.
- Anonymised data use for quality improvement and research (opt-out available).

Separate **Specific Consent** is required for:
- Sharing records with third parties (insurance companies, employers, legal authorities).
- Participation in clinical research studies.
- Photography or video recording (for clinical documentation or teaching).
- Telemedicine consultations (data transmission over the internet).

### 4.2 Patient Rights under PDPA

Patients have the right to:
1. **Access**: Request a copy of their medical records (processed within 14 days; fee: LKR 500 for printed copy, free for digital via the MyNawaloka patient portal).
2. **Rectification**: Request correction of inaccurate data.
3. **Restriction**: Request limitation of processing in certain circumstances.
4. **Data Portability**: Receive their data in a structured, machine-readable format (HL7 FHIR).
5. **Objection**: Object to data processing for direct marketing or research (opt-out).
6. **Erasure**: Request deletion of non-essential data (subject to legal retention requirements).

### 4.3 Record Retention

| Record Type | Retention Period |
|-------------|-----------------|
| Adult patient medical records | 10 years after last visit |
| Paediatric patient records | Until age 25 or 10 years after last visit (whichever is later) |
| Surgical / anaesthesia records | 10 years |
| Radiology images | 7 years |
| Laboratory results | 7 years |
| Billing records | 7 years |
| Employee health records | 30 years |

After the retention period, records are securely destroyed (digital: certified data wiping; physical: cross-cut shredding by an approved vendor).

## 5. Information Security Measures

### 5.1 Network Security

- **Firewall & IDS/IPS**: Fortinet next-generation firewall with intrusion detection/prevention at the network perimeter.
- **Network Segmentation**: Clinical network (HIS, PACS, medical devices) is logically separated from the corporate network (email, internet) and the guest Wi-Fi network.
- **VPN**: All remote access to hospital systems requires a VPN connection with 2FA.
- **Wi-Fi**: Staff Wi-Fi uses WPA3-Enterprise with certificate-based authentication. Guest Wi-Fi is isolated and bandwidth-limited.

### 5.2 Endpoint Security

- All hospital-owned devices (desktops, laptops, tablets) have:
  - Full-disk encryption (BitLocker / FileVault).
  - Endpoint Detection and Response (EDR) software (CrowdStrike Falcon).
  - Automatic OS and application patching (monthly cycle).
- Personal devices (BYOD) are permitted for email only via the MDM (Mobile Device Management) solution; no patient data may be stored on personal devices.

### 5.3 Email & Communication

- Patient-identifiable information must **never** be sent via unencrypted email.
- Internal clinical communication uses the hospital's secure messaging platform (integrated into the HIS).
- WhatsApp, Viber, and other consumer messaging apps are **prohibited** for sharing patient data (even in de-identified form).

### 5.4 Physical Security

- Server rooms are access-controlled (biometric + PIN) and monitored by CCTV.
- Printed patient records are stored in locked cabinets within the Medical Records Department.
- Paper documents awaiting digitisation are kept in secure transit trolleys.
- Clean-desk policy: No patient data visible on unattended desks.

## 6. Data Breach Response

### 6.1 Definition

A data breach is any event where patient data or confidential hospital data is:
- Accessed by an unauthorised person.
- Lost, stolen, or destroyed without authorisation.
- Disclosed to an unintended recipient.

### 6.2 Response Steps

1. **Contain**: Immediately isolate the affected system or revoke the compromised credential.
2. **Report**: Notify the IT Security Team (ext. 2222) and the Privacy Officer within 1 hour.
3. **Assess**: Determine the scope, affected individuals, and data types.
4. **Notify**: If the breach affects > 100 individuals or involves Highly Confidential data, notify the Data Protection Authority within 72 hours and affected individuals "without undue delay."
5. **Remediate**: Implement corrective actions (patch, access review, policy update).
6. **Document**: Complete the Data Breach Report (Form SEC-01) within 7 days.

### 6.3 Annual Drill

A simulated data breach exercise is conducted annually by the IT Security Team to test the response plan and train staff.

## 7. Staff Training & Awareness

- All new employees complete a **1-hour Data Privacy & Security e-learning module** within the first week of employment.
- Annual refresher training is mandatory for all staff (completion tracked via the LMS).
- Department-specific training is provided for roles handling Highly Confidential data (e.g., psychiatry, HIV clinic, genetics).
- Phishing simulation campaigns are conducted quarterly; staff who click on simulated phishing links receive immediate remedial training.

---

*Last Revised: January 2026 — Information Security & Privacy Office, Nawaloka Hospitals PLC*
