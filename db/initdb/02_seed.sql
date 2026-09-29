INSERT INTO regions (region_code, region_name, description)
VALUES ('BUSAN', '부산광역시', 'TripFit 리뷰 분석 대상 지역')
ON CONFLICT (region_code) DO NOTHING;

INSERT INTO place_categories (category_code, category_name)
VALUES ('hotel', '호텔'), ('restaurant', '식당'), ('attraction', '관광지')
ON CONFLICT (category_code) DO NOTHING;

INSERT INTO companion_types (companion_code, companion_name)
VALUES
  ('solo', '혼자'), ('couple', '연인'), ('friends', '친구'),
  ('parents', '부모님'), ('family_with_kids', '아이 동반 가족')
ON CONFLICT (companion_code) DO NOTHING;
