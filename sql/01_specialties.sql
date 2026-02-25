-- specialties: 10 rows
-- Exported from Supabase for deterministic seeding

-- Clear existing data
TRUNCATE TABLE specialties CASCADE;

INSERT INTO specialties (specialty_id, name) VALUES ('065cb3a8-0b15-4f24-b345-2218b13e8021', 'Cardiology');
INSERT INTO specialties (specialty_id, name) VALUES ('4943d315-4f01-48a3-b250-5d5c44627504', 'Gynecology');
INSERT INTO specialties (specialty_id, name) VALUES ('54475d5d-41fb-4381-97b7-e532b7aa821b', 'Radiology');
INSERT INTO specialties (specialty_id, name) VALUES ('64957470-765a-482d-b0e2-4851f18d3e67', 'Neurology');
INSERT INTO specialties (specialty_id, name) VALUES ('7684faab-bb76-4cd6-b02b-c994b2b6af30', 'Pediatrics');
INSERT INTO specialties (specialty_id, name) VALUES ('7ec197a0-151e-48e2-92dd-9b31075a8cfa', 'General Practice');
INSERT INTO specialties (specialty_id, name) VALUES ('90cf5e51-c4f2-4be1-94a0-c5a578fe758e', 'Orthopedics');
INSERT INTO specialties (specialty_id, name) VALUES ('c5cbf332-956a-477e-9044-2a1a573e9e5f', 'Dermatology');
INSERT INTO specialties (specialty_id, name) VALUES ('f22cfdf2-b23d-436d-b104-e511afd749a1', 'Psychiatry');
INSERT INTO specialties (specialty_id, name) VALUES ('f76a7665-450e-4d4d-9ce5-b175db2bf9ac', 'Pathology');

-- End of specialties
