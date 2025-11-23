CREATE DATABASE IF NOT EXISTS ragodb
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;
USE ragodb;

-- 일반 대화 메시지
CREATE TABLE IF NOT EXISTS gen_message (
  id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  gen_chat_id VARCHAR(100) NOT NULL,
  sender_type ENUM('user','assistant','system') DEFAULT 'user',
  content TEXT NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_gen_chat_id_created_at (gen_chat_id, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 무역(문서) 대화 메시지
CREATE TABLE IF NOT EXISTS doc_message (
  id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  trade_id VARCHAR(100) NOT NULL,
  sender_type ENUM('user','assistant','system') DEFAULT 'user',
  content TEXT NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_trade_id_created_at (trade_id, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 일반 대화 요약
CREATE TABLE IF NOT EXISTS gen_chat_summary (
  id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  gen_chat_id VARCHAR(100) NOT NULL,
  summary TEXT NOT NULL,
  message_count INT NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_gen_summary (gen_chat_id, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 무역 플로우 요약
CREATE TABLE IF NOT EXISTS trade_flow_summary (
  id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  trade_id VARCHAR(100) NOT NULL,
  summary TEXT NOT NULL,
  message_count INT NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_trade_summary (trade_id, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 무역 플로우 메타데이터
CREATE TABLE IF NOT EXISTS trade_flow (
  trade_id VARCHAR(100) PRIMARY KEY,
  title VARCHAR(255) NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 문서 템플릿
CREATE TABLE IF NOT EXISTS doc_template (
  template_id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  template_name VARCHAR(100) NOT NULL UNIQUE,
  description VARCHAR(255),
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 문서
CREATE TABLE IF NOT EXISTS document (
  doc_id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  trade_id VARCHAR(100) NOT NULL,
  template_id BIGINT UNSIGNED,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_document_trade (trade_id),
  CONSTRAINT fk_document_trade
    FOREIGN KEY (trade_id) REFERENCES trade_flow(trade_id)
    ON DELETE CASCADE,
  CONSTRAINT fk_document_template
    FOREIGN KEY (template_id) REFERENCES doc_template(template_id)
    ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 문서 버전
CREATE TABLE IF NOT EXISTS doc_version (
  version_id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  doc_id BIGINT UNSIGNED NOT NULL,
  title VARCHAR(255),
  content LONGTEXT,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_doc_version (doc_id, created_at),
  CONSTRAINT fk_doc_version_doc
    FOREIGN KEY (doc_id) REFERENCES document(doc_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;