import Foundation

// MARK: - Codable extensions for FFI types (from memme_ffi.swift)

extension MemoryResult: Codable {
    enum CodingKeys: String, CodingKey {
        case id, content, score, metadata, importance, categories, immutable, retention, stability
        case userId = "user_id"
        case agentId = "agent_id"
        case appId = "app_id"
        case runId = "run_id"
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case accessCount = "access_count"
        case expirationDate = "expiration_date"
        case eventTime = "event_time"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            id: try c.decode(String.self, forKey: .id),
            content: try c.decode(String.self, forKey: .content),
            userId: try c.decode(String.self, forKey: .userId),
            agentId: try c.decodeIfPresent(String.self, forKey: .agentId),
            appId: try c.decodeIfPresent(String.self, forKey: .appId),
            runId: try c.decodeIfPresent(String.self, forKey: .runId),
            score: try c.decodeIfPresent(Float.self, forKey: .score),
            createdAt: try c.decodeIfPresent(String.self, forKey: .createdAt) ?? "",
            updatedAt: try c.decodeIfPresent(String.self, forKey: .updatedAt) ?? "",
            metadata: try c.decodeIfPresent(String.self, forKey: .metadata),
            importance: try c.decodeIfPresent(Float.self, forKey: .importance) ?? 0.5,
            accessCount: try c.decodeIfPresent(UInt32.self, forKey: .accessCount) ?? 0,
            immutable: try c.decodeIfPresent(Bool.self, forKey: .immutable) ?? false,
            expirationDate: try c.decodeIfPresent(String.self, forKey: .expirationDate),
            categories: try c.decodeIfPresent([String].self, forKey: .categories) ?? [],
            retention: try c.decodeIfPresent(Float.self, forKey: .retention) ?? 1.0,
            stability: try c.decodeIfPresent(Float.self, forKey: .stability) ?? 1.0,
            eventTime: try c.decodeIfPresent(String.self, forKey: .eventTime)
        )
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(content, forKey: .content)
        try c.encode(userId, forKey: .userId)
        try c.encodeIfPresent(agentId, forKey: .agentId)
        try c.encodeIfPresent(appId, forKey: .appId)
        try c.encodeIfPresent(runId, forKey: .runId)
        try c.encodeIfPresent(score, forKey: .score)
        try c.encode(createdAt, forKey: .createdAt)
        try c.encode(updatedAt, forKey: .updatedAt)
        try c.encodeIfPresent(metadata, forKey: .metadata)
        try c.encode(importance, forKey: .importance)
        try c.encode(accessCount, forKey: .accessCount)
        try c.encode(immutable, forKey: .immutable)
        try c.encodeIfPresent(expirationDate, forKey: .expirationDate)
        try c.encode(categories, forKey: .categories)
        try c.encode(retention, forKey: .retention)
        try c.encode(stability, forKey: .stability)
        try c.encodeIfPresent(eventTime, forKey: .eventTime)
    }
}

extension MemoryResult: Identifiable {}

extension Entity: Codable {
    enum CodingKeys: String, CodingKey {
        case id, name
        case entityType = "entity_type"
        case userId = "user_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            id: try c.decode(String.self, forKey: .id),
            name: try c.decode(String.self, forKey: .name),
            entityType: try c.decodeIfPresent(String.self, forKey: .entityType),
            userId: try c.decodeIfPresent(String.self, forKey: .userId) ?? ""
        )
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(name, forKey: .name)
        try c.encodeIfPresent(entityType, forKey: .entityType)
        try c.encode(userId, forKey: .userId)
    }
}

extension Entity: Identifiable {}

extension GraphRelation: Codable {
    enum CodingKeys: String, CodingKey {
        case id, source, target
        case relationType = "relation_type"
        case sourceId = "source_id"
        case targetId = "target_id"
        case userId = "user_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            id: try c.decodeIfPresent(String.self, forKey: .id) ?? UUID().uuidString,
            source: try c.decode(String.self, forKey: .source),
            sourceId: try c.decodeIfPresent(String.self, forKey: .sourceId) ?? "",
            target: try c.decode(String.self, forKey: .target),
            targetId: try c.decodeIfPresent(String.self, forKey: .targetId) ?? "",
            relationType: try c.decode(String.self, forKey: .relationType),
            userId: try c.decodeIfPresent(String.self, forKey: .userId) ?? ""
        )
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encodeIfPresent(id, forKey: .id)
        try c.encode(source, forKey: .source)
        try c.encode(sourceId, forKey: .sourceId)
        try c.encode(target, forKey: .target)
        try c.encode(targetId, forKey: .targetId)
        try c.encode(relationType, forKey: .relationType)
        try c.encode(userId, forKey: .userId)
    }
}

extension GraphRelation: Identifiable {}

extension GraphSearchResult: Codable {
    enum CodingKeys: String, CodingKey { case entities, relations }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            entities: try c.decode([Entity].self, forKey: .entities),
            relations: try c.decode([GraphRelation].self, forKey: .relations)
        )
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(entities, forKey: .entities)
        try c.encode(relations, forKey: .relations)
    }
}

extension UserStats: Codable {
    enum CodingKeys: String, CodingKey {
        case userId = "user_id"
        case totalMemories = "total_memories"
        case totalEntities = "total_entities"
        case totalRelationships = "total_relationships"
        case earliestMemory = "earliest_memory"
        case latestMemory = "latest_memory"
        case uniqueAgents = "unique_agents"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            userId: try c.decode(String.self, forKey: .userId),
            totalMemories: try c.decode(UInt64.self, forKey: .totalMemories),
            totalEntities: try c.decode(UInt64.self, forKey: .totalEntities),
            totalRelationships: try c.decode(UInt64.self, forKey: .totalRelationships),
            earliestMemory: try c.decodeIfPresent(String.self, forKey: .earliestMemory),
            latestMemory: try c.decodeIfPresent(String.self, forKey: .latestMemory),
            uniqueAgents: try c.decodeIfPresent(UInt64.self, forKey: .uniqueAgents) ?? 0
        )
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(userId, forKey: .userId)
        try c.encode(totalMemories, forKey: .totalMemories)
        try c.encode(totalEntities, forKey: .totalEntities)
        try c.encode(totalRelationships, forKey: .totalRelationships)
        try c.encodeIfPresent(earliestMemory, forKey: .earliestMemory)
        try c.encodeIfPresent(latestMemory, forKey: .latestMemory)
        try c.encode(uniqueAgents, forKey: .uniqueAgents)
    }
}

extension EntityStat: Codable {
    enum CodingKeys: String, CodingKey {
        case name
        case entityType = "entity_type"
        case relationshipCount = "relationship_count"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            name: try c.decode(String.self, forKey: .name),
            entityType: try c.decodeIfPresent(String.self, forKey: .entityType),
            relationshipCount: try c.decode(UInt64.self, forKey: .relationshipCount)
        )
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(name, forKey: .name)
        try c.encodeIfPresent(entityType, forKey: .entityType)
        try c.encode(relationshipCount, forKey: .relationshipCount)
    }
}

extension EntityStat: Identifiable {
    public var id: String { name }
}

extension TimeBucket: Codable {
    enum CodingKeys: String, CodingKey { case period, count }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            period: try c.decode(String.self, forKey: .period),
            count: try c.decode(UInt64.self, forKey: .count)
        )
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(period, forKey: .period)
        try c.encode(count, forKey: .count)
    }
}

extension TimeBucket: Identifiable {
    public var id: String { period }
}


